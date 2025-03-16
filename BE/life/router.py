import logging
from typing import List


from fastapi import APIRouter, HTTPException
from pydantic import UUID4
# from src.life.schemas import Chunk, SourceResponse
# from src.life.utils import get_message_source
# from src.life.utils import get_source_content as get_source_content_function

router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@router.post(
    "", dependencies=[], response_model=''
)
async def extract_question(QueryText:str):
    try:
        print(f'{QueryText} received from user ')
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
