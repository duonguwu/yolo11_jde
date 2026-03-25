import os
import zipfile
from pathlib import Path
from cvat_sdk import make_client, Client
from cvat_sdk.core.proxies.tasks import Task
from cvat_sdk.core.proxies.jobs import Job
from cvat_sdk.core.proxies.types import Location

from dotenv import load_dotenv


def get_completed_tasks(client: Client) -> list[Task]:
    """
    Fetch all tasks with status 'completed' from a CVAT server.

    Args:
        client_url (str): URL of the CVAT server.
        username (str): CVAT username for login.
        password (str): CVAT password.

    Returns:
        List[Task]: List of completed task objects.
    """

    tasks = client.tasks.list()
    completed = [t for t in tasks if t.status == "completed"]

    return completed


def get_task_have_all_job_completed(client: Client) -> list[Task]:
    """
    Fetch all task have all job completed
    """
    tasks: list[Task] = client.tasks.list()
    completed_tasks: list[Task] = []

    for task in tasks:
        jobs_in_task = task.get_jobs()
        is_all_completed = all(job.state == "completed" for job in jobs_in_task)

        if is_all_completed:
            completed_tasks.append(task)

    return completed_tasks


def export_task_annotations(
    task: Task,
    include_images: bool = False,
    format="COCO 1.0",
    filename: str = "annotations.zip",
    extract_dir: str = "annotations",
    delete_zip: bool = True,
) -> None:
    path = task.export_dataset(
        format_name="COCO 1.0",
        filename=filename,
        location=Location.LOCAL,
        include_images=include_images,
    )

    if path is None:
        raise Exception("Failed to export annotations")

    # unzip file
    with zipfile.ZipFile(str(path), "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    if delete_zip:
        os.remove(path)

    # Special handling for COCO format (rename main json file)
    if format == "COCO 1.0":
        os.rename(
            os.path.join(extract_dir, "annotations/instances_default.json"),
            os.path.join(extract_dir, f"{task.name}_gt.json"),
        )


if __name__ == "__main__":
    load_dotenv()

    CVAT_URL = os.getenv("CVAT_URL")
    CVAT_ADMIN_USER = os.getenv("CVAT_ADMIN_USER")
    CVAT_ADMIN_PWD = os.getenv("CVAT_ADMIN_PWD")

    if not CVAT_URL or not CVAT_ADMIN_USER or not CVAT_ADMIN_PWD:
        raise Exception("CVAT_URL, CVAT_ADMIN_USER, CVAT_ADMIN_PWD must be set")

    with make_client(
        host=CVAT_URL,
        credentials=(CVAT_ADMIN_USER, CVAT_ADMIN_PWD),
    ) as client:
        client.organization_slug = "PersonDetInfo"
        complete_tasks = get_task_have_all_job_completed(client)
        export_task_annotations(complete_tasks[0])
