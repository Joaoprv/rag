from fastapi import APIRouter, UploadFile
from document_analyser.dependecies import SessionDep, DocumentSvcDep
from document_analyser.models import File
from document_analyser.schemas import FileRead
from sqlmodel import select
from uuid import UUID

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload/")
async def upload_file(
    file: UploadFile, name: str, session: SessionDep, document_svc: DocumentSvcDep
) -> UUID:
    db_file = File(name=name, content=await file.read())
    res = await document_svc.save(session, db_file)

    return res.id


@router.get("/", response_model=list[FileRead])
async def get_files(session: SessionDep):
    return session.exec(select(File)).all()