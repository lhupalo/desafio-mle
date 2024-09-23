from fastapi import APIRouter

router = APIRouter()


@router.get("/health", status_code=200, tags=["health"], summary="Health check")
async def health():
    return {"status": "ok"}
