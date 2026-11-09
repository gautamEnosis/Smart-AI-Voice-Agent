"""Integration tests for the Phase 5 administrative HTTP contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.knowledge import KnowledgeHubService
from backend.api.runtime import ProviderRuntimeStore
from backend.config.settings import AppSettings, get_settings
from backend.rag.vector_store import VectorStoreError


class TestAdminAPI(unittest.TestCase):
    """Test administrative HTTP endpoints under authentication."""

    def test_admin_api_hides_credentials_and_accepts_safe_document_upload(self) -> None:
        """The browser sees safe provider status and can index a tenant document when authenticated."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            tmp_path = Path(temp_dir)
            settings = AppSettings(
                chroma_persist_dir=str(tmp_path / "chroma"),
                admin_api_key="dev-admin-key",
            )
            runtime = ProviderRuntimeStore(settings, state_path=tmp_path / "provider-runtime.json")
            knowledge_hub = KnowledgeHubService(settings, data_dir=tmp_path / "knowledge")

            with (
                patch("backend.api.routes.admin.get_provider_runtime_store", return_value=runtime),
                patch("backend.api.routes.admin.get_knowledge_hub", return_value=knowledge_hub),
            ):
                client = TestClient(create_app())
                client.app.dependency_overrides[get_settings] = lambda: settings
                auth_headers = {"Authorization": "Bearer dev-admin-key"}
                providers = client.get("/api/v1/admin/providers", headers=auth_headers)
                upload = client.post(
                    "/api/v1/admin/knowledge/upload",
                    data={"tenant_id": "grand-meridian"},
                    files={"document": ("arrival.md", b"Check-in begins at 3 PM.", "text/markdown")},
                    headers=auth_headers,
                )
                dashboard = client.get("/api/v1/admin/dashboard", headers=auth_headers)

            self.assertEqual(providers.status_code, 200)
            self.assertNotIn("api_key", providers.text)
            self.assertEqual(upload.status_code, 200)
            self.assertEqual(upload.json()["chunks_indexed"], 1)
            self.assertEqual(dashboard.status_code, 200)
            knowledge_health = next(
                service for service in dashboard.json()["services"] if service["name"] == "Knowledge retrieval"
            )
            self.assertEqual(knowledge_health["status"], "ready")

            reloaded_hub = KnowledgeHubService(settings, data_dir=tmp_path / "knowledge")
            result = reloaded_hub.search("grand-meridian", "When does check-in begin?")
            self.assertTrue(result.has_context)
            self.assertIn("3 PM", result.chunks[0].excerpt)

    def test_admin_api_rejects_unauthenticated_requests(self) -> None:
        """Admin endpoints reject requests without a valid Bearer token."""
        client = TestClient(create_app())
        response = client.get("/api/v1/admin/providers")
        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing or malformed", response.json()["detail"])

    def test_knowledge_search_reports_storage_failure_instead_of_no_match(self) -> None:
        """A vector-store failure is an actionable service error, not empty search results."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            settings = AppSettings(admin_api_key="dev-admin-key")
            service = KnowledgeHubService(settings, data_dir=Path(temp_dir) / "knowledge")
            failing_store = MagicMock()
            failing_store.search.side_effect = VectorStoreError("ChromaDB query failed.")

            with (
                patch("backend.api.routes.admin.get_knowledge_hub", return_value=service),
                patch("backend.api.knowledge.get_vector_store", return_value=failing_store),
            ):
                client = TestClient(create_app())
                client.app.dependency_overrides[get_settings] = lambda: settings
                response = client.post(
                    "/api/v1/admin/knowledge/search",
                    json={"tenant_id": "grand-meridian", "query": "What time is check-in?"},
                    headers={"Authorization": f"Bearer {settings.admin_api_key}"},
                )

            self.assertEqual(response.status_code, 503)
            self.assertIn("ChromaDB query failed", response.json()["detail"])
