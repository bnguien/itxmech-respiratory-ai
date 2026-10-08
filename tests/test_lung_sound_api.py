import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import wave

import httpx
from fastapi.testclient import TestClient

from app.main import app, settings
from app.schemas.lung import LungSoundAnalyzeResponse


class BrokenDownload(httpx.SyncByteStream):
    def __iter__(self):
        yield b"partial audio"
        raise httpx.ReadError("Sensitive download details")


class LungSoundAPITest(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "recording_id": "76f05e2b-210c-47ce-8704-e641c4c25c64",
            "audio_url": "https://storage.example.com/audio.wav?token=secret",
        }
        self.endpoint = f"{settings.api_prefix}/lung-sound/analyze"
        self.result = {
            "boundaries": [0.0, 1.0, 1.05],
            "cycles": [
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "duration": 1.0,
                    "status": "processed",
                    "label": "normal",
                    "confidence": 0.9,
                    "probabilities": {
                        "normal": 0.9, "crackle": 0.05,
                        "wheeze": 0.03, "both": 0.02,
                    },
                    "n_frames": 98,
                },
                {
                    "index": 1,
                    "start": 1.0,
                    "end": 1.05,
                    "duration": 0.05,
                    "status": "unprocessable",
                    "reason": "Cycle has too few Fbank frames.",
                    "label": None,
                    "confidence": None,
                    "probabilities": None,
                },
            ],
        }
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000)
        self.audio = buffer.getvalue()
        self.pipeline = Mock()
        self.pipeline.analyze_file.side_effect = self.analyze_file
        self.factory = self.enterContext(
            patch("app.main.LungSoundPipeline", return_value=self.pipeline)
        )
        self.client = self.enterContext(TestClient(app))
        self.directories = []
        self.enterContext(patch(
            "app.api.lung_sound.TemporaryDirectory",
            side_effect=self.temporary_directory,
        ))
        self.requests = []
        self.download_response = lambda: httpx.Response(200, content=self.audio)
        real_client = httpx.Client
        self.download_client = self.enterContext(patch(
            "app.api.lung_sound.httpx.Client",
            side_effect=lambda **kwargs: real_client(
                transport=httpx.MockTransport(self.download), **kwargs
            ),
        ))

    def temporary_directory(self):
        directory = TemporaryDirectory()
        self.directories.append(Path(directory.name))
        return directory

    def download(self, request):
        self.requests.append(request)
        return self.download_response()

    def analyze_file(self, wav_path):
        self.assertEqual(wav_path.suffix, ".wav")
        self.assertEqual(wav_path.read_bytes(), self.audio)
        return self.result

    def assert_cleaned_up(self):
        self.assertTrue(self.directories)
        for directory in self.directories:
            self.assertFalse(directory.exists())

    def test_success_and_single_pipeline_initialization(self):
        for _ in range(2):
            response = self.client.post(self.endpoint, json=self.payload)
            self.assertEqual(response.status_code, 200)
            expected = LungSoundAnalyzeResponse(
                recording_id=self.payload["recording_id"], **self.result
            ).model_dump(mode="json")
            self.assertEqual(response.json(), expected)
            self.assertIsNone(response.json()["cycles"][0]["reason"])
            self.assertIsNone(response.json()["cycles"][1]["n_frames"])
        self.factory.assert_called_once_with()
        self.assertEqual(self.pipeline.analyze_file.call_count, 2)
        self.assertEqual(str(self.requests[0].url), self.payload["audio_url"])
        self.assert_cleaned_up()

    def test_download_http_failure(self):
        for status in (403, 404, 500):
            with self.subTest(status=status):
                self.download_response = lambda: httpx.Response(status)
                response = self.client.post(self.endpoint, json=self.payload)
                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.json(), {"detail": "Unable to download audio."})
                self.assert_cleaned_up()
        self.pipeline.analyze_file.assert_not_called()

    def test_download_timeout(self):
        self.download_response = Mock(side_effect=httpx.ReadTimeout("secret URL"))
        response = self.client.post(self.endpoint, json=self.payload)
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret", response.text)
        self.pipeline.analyze_file.assert_not_called()
        self.assert_cleaned_up()

    def test_partial_download_cleanup(self):
        self.download_response = lambda: httpx.Response(200, stream=BrokenDownload())
        response = self.client.post(self.endpoint, json=self.payload)
        self.assertEqual(response.status_code, 502)
        self.pipeline.analyze_file.assert_not_called()
        self.assert_cleaned_up()

    def test_invalid_payload(self):
        for payload in (
            {},
            {**self.payload, "recording_id": "not-a-uuid"},
            {**self.payload, "audio_url": "not-a-url"},
            {**self.payload, "audio_url": "file:///private/audio.wav"},
            {"recording_id": self.payload["recording_id"]},
        ):
            with self.subTest(payload=payload):
                response = self.client.post(self.endpoint, json=payload)
                self.assertEqual(response.status_code, 422)
        self.download_client.assert_not_called()
        self.pipeline.analyze_file.assert_not_called()
        self.assertEqual(self.directories, [])

    def test_pipeline_exception_cleanup(self):
        def fail(wav_path):
            self.assertEqual(wav_path.read_bytes(), self.audio)
            raise RuntimeError("Sensitive model path and stack details")

        self.pipeline.analyze_file.side_effect = fail
        response = self.client.post(self.endpoint, json=self.payload)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Unable to analyze audio."})
        self.assert_cleaned_up()

    def test_health_and_docs(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "respicare-ai"})
        self.assertEqual(self.client.get("/docs").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        operation = schema["paths"][self.endpoint]["post"]
        response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
        self.assertEqual(
            response_schema["$ref"],
            "#/components/schemas/LungSoundAnalyzeResponse",
        )


if __name__ == "__main__":
    unittest.main()
