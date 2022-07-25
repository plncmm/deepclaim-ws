import fastapi

tags_metadata = [
    ...
]

app = fastapi.FastAPI(title="DeepClaim", description="API para comunicarse con el servicio de predicción de DeepClaim.", openapi_tags=tags_metadata)