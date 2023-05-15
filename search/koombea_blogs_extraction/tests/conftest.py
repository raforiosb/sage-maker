from typing import Generator
from koombea_blogs.db.deps import get_connection
from koombea_blogs.config.settings import settings
import wandb
import pytest


@pytest.fixture(scope="module")
def conn() -> Generator:
    # set up test app
    for conn in get_connection():
        # testing
        yield conn
    # tear down


@pytest.fixture(scope="module")
def wandb_run() -> Generator:
    # set up test wandb run
    with wandb.init(
        id="test_id",
        project="test-integration-project",
        entity=settings.WANDB_ENTITY,
        job_type="test-job",
        tags=["test"],
        name="test-run",
        notes="this is just a simple test run",
        resume=True,
    ) as wandb_run:
        yield wandb_run
