import fastapi
import pydantic

import deepclaim_ws.data
import deepclaim_ws.dummy_classifier
import typing
import passlib.context
import datetime
import jose.jwt
import jose
import dotenv
import json

env = dotenv.dotenv_values("secret.env")

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = env["SECRET_KEY"]
ALGORITHM = env["ALGORITHM"]
ACCESS_TOKEN_EXPIRE_MINUTES = 30

with open('users.json') as f:
    clients_db = json.load(f)

class Token(pydantic.BaseModel):
    access_token: str
    token_type: str

class TokenData(pydantic.BaseModel):
    client_id: typing.Union[str, None] = None


class Client(pydantic.BaseModel):
    client_id: str
    disabled: typing.Union[bool, None] = None


class ClientInDB(Client):
    hashed_client_secret: str
    
class ClientIn(pydantic.BaseModel):
    client_id: str
    client_secret: str


pwd_context = passlib.context.CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = fastapi.security.OAuth2PasswordBearer(tokenUrl="token2")


app = fastapi.FastAPI(title="DeepClaim", description="API para comunicarse con el servicio de predicción de DeepClaim.")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)

def get_client(db, client_id: str):
    if client_id in db:
        user_dict = db[client_id]
        return ClientInDB(**user_dict)


def authenticate_user(db, client_id: str, client_secret: str):
    client = get_client(db, client_id)
    if not client:
        return False
    if not verify_password(client_secret, client.hashed_client_secret):
        return False
    return client


def create_access_token(data: dict, expires_delta: typing.Union[datetime.timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jose.jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_client(token: str = fastapi.Depends(oauth2_scheme)):
    credentials_exception = fastapi.HTTPException(
        status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jose.jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        client_id: str = payload.get("sub")
        if client_id is None:
            raise credentials_exception
        token_data = TokenData(client_id=client_id)
    except jose.JWTError:
        raise credentials_exception
    user = get_client(clients_db, client_id=token_data.client_id)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_client(current_client: Client = fastapi.Depends(get_current_client)):
    if current_client.disabled:
        raise fastapi.HTTPException(status_code=400, detail="Inactive user")
    return current_client

@app.post("/token2", response_model=Token, include_in_schema=False)
async def login_for_access_token(form_data: fastapi.security.OAuth2PasswordRequestForm = fastapi.Depends()):
    user = authenticate_user(clients_db, form_data.username, form_data.password)
    if not user:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.client_id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", response_model=deepclaim_ws.data.Token)
async def login_for_access_token(form_data: ClientIn = fastapi.Depends()):
    client = authenticate_user(clients_db, form_data.client_id, form_data.client_secret)
    if not client:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": client.client_id}, expires_delta=access_token_expires
    )
    response = deepclaim_ws.data.Token(
        TransactionID=0,
        Status = deepclaim_ws.data.StatusClass(
            Code = 200,
            Value = "OK"
        ),
        accessToken=access_token
    )
    return response

async def checker(datos_reclamo: str = fastapi.Form(...)):
    try:
        model = deepclaim_ws.data.JsonReclamo.parse_raw(datos_reclamo)
        return model
    except pydantic.ValidationError as e:
        raise fastapi.exceptions.HTTPException(detail=fastapi.encoders.jsonable_encoder(e.errors()), status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.post("/reclamo", response_model=deepclaim_ws.data.ReclamoOut, summary="Clasificar reclamo.", description="Este servicio recibe los datos del reclamo.", dependencies=[fastapi.Depends(get_current_active_client)])
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
        Data = deepclaim_ws.data.RespuestaClass(
            id_caso = datos_reclamo.id_caso,
            mercado = deepclaim_ws.data.MercadoClass(
                tipo_mercado = c_mercado.predict([datos_reclamo.descripcion_problema])[0],
                tipo_entidad = c_tipo_entidad.predict([datos_reclamo.descripcion_problema])[0],
                nombre_entidad = c_nombre_entidad.predict([datos_reclamo.descripcion_problema])[0],
                tipo_producto = c_tipo_producto.predict([datos_reclamo.descripcion_problema])[:2],
                tipo_materia = c_tipo_materia.predict([datos_reclamo.descripcion_problema])[:2]
            )
        )
    )
    return response

@app.post("/reclamo_diario", response_model=deepclaim_ws.data.ReclamoDiarioOut, summary="Clasificar reclamos diarios.", description="Este servicio recibe los datos del reclamo ingresado por el ciudadano con los datos del día anterior con el fin de poder realizar cuadraturas al servicio.", dependencies=[fastapi.Depends(get_current_active_client)])
async def classify_reclamo_diario(
    datos_diario: deepclaim_ws.data.JsonReclamos
    ):  
    response = deepclaim_ws.data.ReclamoDiarioOut(
        TransactionID=0,
        Status = deepclaim_ws.data.StatusClass(
            Code = 200,
            Value = "OK"
        )
    )
    return response

@app.post("/reclamo_sinRespuesta", response_model=deepclaim_ws.data.ReclamoOut, summary="Clasificar reclamo.", description="Este servicio recibe los datos del reclamo que no han tenido clasificación por su equipo, este cliente se activará 4 veces al día y enviará los datos del reclamocomo si se tratará del servicio Reclamo pero de manera diferida", dependencies=[fastapi.Depends(get_current_active_client)])
async def classify_reclamo_sinRespuesta(
    datos_reclamo: deepclaim_ws.data.Json = fastapi.Depends(checker), 
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
        Data = deepclaim_ws.data.RespuestaClass(
            id_caso = datos_reclamo.id_caso,
            mercado = deepclaim_ws.data.MercadoClass(
                tipo_mercado = c_mercado.predict([datos_reclamo.descripcion_problema])[0],
                tipo_entidad = c_tipo_entidad.predict([datos_reclamo.descripcion_problema])[0],
                nombre_entidad = c_nombre_entidad.predict([datos_reclamo.descripcion_problema])[0],
                tipo_producto = c_tipo_producto.predict([datos_reclamo.descripcion_problema])[:2],
                tipo_materia = c_tipo_materia.predict([datos_reclamo.descripcion_problema])[:2]
            )
        )
    )
    return response

@app.post("/reclamo_retro", response_model=deepclaim_ws.data.ReclamoRetroOut, summary="Clasificar reclamos diarios.", description="Este servicio recibe los datos del reclamo con la clasificación ingresada por los analistas de CMF, con el fin de que puedan realizar retroalimentación a la clasificación proporcionada por el algoritmo.", dependencies=[fastapi.Depends(get_current_active_client)])
async def receive_reclamo_retro(
    datos_clasificacion: deepclaim_ws.data.JsonClasificacion
    ):  
    response = deepclaim_ws.data.ReclamoRetroOut(
        TransactionID=0,
        Status = deepclaim_ws.data.StatusClass(
            Code = 200,
            Value = "OK"
        )
    )
    return response