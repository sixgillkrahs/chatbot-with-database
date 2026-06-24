from fastapi import APIRouter

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)


@router.get("/")
def read_root():
    return {"Hello": "World"}
