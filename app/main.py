import fastapi
import pydantic

import deepclaim_ws.data
import deepclaim_ws.dummy_classifier
import typing

app = fastapi.FastAPI(title="DeepClaim", description="API para comunicarse con el servicio de predicción de DeepClaim.")

async def checker(datos_reclamo: str = fastapi.Form(...)):
    try:
        model = deepclaim_ws.data.JsonReclamo.parse_raw(datos_reclamo)
        return model
    except pydantic.ValidationError as e:
        raise fastapi.exceptions.HTTPException(detail=fastapi.encoders.jsonable_encoder(e.errors()), status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.post("/reclamo", response_model=deepclaim_ws.data.ReclamoOut, summary="Clasificar reclamo.", description="Este servicio recibe los datos del reclamo.")
async def classify_reclamo(
    datos_reclamo: deepclaim_ws.data.JsonReclamo = fastapi.Depends(checker), 
    archivo_1: typing.Optional[fastapi.UploadFile] = fastapi.File(None), 
    archivo_2: typing.Optional[fastapi.UploadFile] = fastapi.File(None)
    ):  
    c_mercado = deepclaim_ws.dummy_classifier.DummyMercadoClassifier()
    c_tipo_entidad = deepclaim_ws.dummy_classifier.DummyTipoEntidadClassifier()
    c_tipo_materia = deepclaim_ws.dummy_classifier.DummyTipoMateriaClassifier()
    c_tipo_producto = deepclaim_ws.dummy_classifier.DummyTipoProductoClassifier()
    c_nombre_entidad = deepclaim_ws.dummy_classifier.DummyNombreEntidadClassifier()
    response = deepclaim_ws.data.ReclamoOut(
        TransactionID=0,
        Status = deepclaim_ws.data.StatusClass(
            Code = 200,
            Value = "OK"
        ),
        Respuesta = deepclaim_ws.data.RespuestaClass(
            Id_caso = datos_reclamo.id_caso,
            Mercado = c_mercado.predict([datos_reclamo.descripcion_problema])[0],
            Tipo_entidad = c_tipo_entidad.predict([datos_reclamo.descripcion_problema])[0],
            Nombre_entidad = c_nombre_entidad.predict([datos_reclamo.descripcion_problema])[0],
            Tipo_producto = c_tipo_producto.predict([datos_reclamo.descripcion_problema])[0],
            Tipo_materia = c_tipo_materia.predict([datos_reclamo.descripcion_problema])[0]
        )
    )
    return response

@app.post("/reclamo_diario", response_model=deepclaim_ws.data.ReclamoDiarioOut, summary="Clasificar reclamos diarios.", description="Este servicio recibe los datos del reclamo ingresado por el ciudadano con los datos del día anterior con el fin de poder realizar cuadraturas al servicio.")
async def classify_reclamo_diario(
    datos_reclamo: deepclaim_ws.data.JsonReclamos
    ):  
    response = deepclaim_ws.data.ReclamoDiarioOut(
        TransactionID=0,
        Status = deepclaim_ws.data.StatusClass(
            Code = 200,
            Value = "OK"
        )
    )
    return response