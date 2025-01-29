from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import logging
from src.preprocessing.cleaner import RoomNameProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.matching.similarity import RoomMatcher
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("api")

app = FastAPI(
    title="Room Matching API",
    description="API for matching hotel room descriptions across different suppliers",
    version="1.0.0"
)

# Initialize components
room_processor = RoomNameProcessor()
feature_extractor = FeatureExtractor()
matcher = RoomMatcher()


class RoomInfo(BaseModel):
    room_id: str
    room_name: str


class RoomMatchRequest(BaseModel):
    reference_rooms: List[RoomInfo]
    supplier_rooms: List[RoomInfo]

    model_config = {
        "json_schema_extra": {
            "example": {
                "reference_rooms": [
                    {"room_id": "ref1", "room_name": "Deluxe King Room with City View"}
                ],
                "supplier_rooms": [
                    {"room_id": "sup1", "room_name": "King Deluxe City View Room"}
                ]
            }
        }
    }


class RoomMatch(BaseModel):
    reference_rooms: List[Dict[str, str]]
    supplier_rooms: List[Dict[str, str]]

    model_config = {
        "json_schema_extra": {
            "example": {
                "reference_rooms": [
                    {"room_name": "Deluxe King Room with City View"}
                ],
                "supplier_rooms": [
                    {"room_name": "King Deluxe City View Room"}
                ]
            }
        }
    }


@app.post("/process_rooms",
          response_model=Dict,
          tags=["Room Matching"],
          summary="Process and match room descriptions"
          )
async def process_rooms(room_data: RoomMatch):
    try:
        logger.info("Processing room descriptions")
        results = []
        for ref_room in room_data.reference_rooms:
            processed_ref = room_processor.process_room_name(ref_room['room_name'])
            features_ref = feature_extractor.extract_features(processed_ref['cleaned_name'])
            results.append({
                'original_name': processed_ref['original_name'],
                'cleaned_name': processed_ref['cleaned_name'],
                'features': features_ref
            })
        logger.info(f"Successfully processed {len(results)} rooms")
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Error processing rooms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match_rooms",
          response_model=Dict,
          tags=["Room Matching"],
          summary="Match rooms between reference and supplier catalogs"
          )
async def match_rooms(request: RoomMatchRequest):
    try:
        logger.info(
            f"Processing match request with {len(request.reference_rooms)} reference rooms "
            f"and {len(request.supplier_rooms)} supplier rooms"
        )

        processed_ref_rooms = [
            {**room_processor.process_room_name(r.room_name),
             'features': feature_extractor.extract_features(r.room_name),
             'id': r.room_id}
            for r in request.reference_rooms
        ]

        processed_supp_rooms = [
            {**room_processor.process_room_name(s.room_name),
             'features': feature_extractor.extract_features(s.room_name),
             'id': s.room_id}
            for s in request.supplier_rooms
        ]

        results = matcher.match_rooms(processed_ref_rooms, processed_supp_rooms)

        logger.info(f"Successfully matched {len(results['matched_pairs'])} room pairs")

        return {
            "status": "success",
            "matches": results["matched_pairs"],
            "unmatched_reference": results["unmatched_reference"],
            "unmatched_supplier": results["unmatched_supplier"]
        }
    except Exception as e:
        logger.error(f"Error in match_rooms endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health",
         tags=["Health"],
         summary="Check API health",
         response_model=Dict[str, str]
         )
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }