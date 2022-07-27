import fastapi
import pydantic
import datetime
import typing

class JsonReclamo(pydantic.BaseModel):
    Id_caso: int = fastapi.Query(default=None, description="ID único interno del ticket de atención")
    fecha_ingreso: datetime.date = fastapi.Query(default=None, description="Fecha ingreso")
    descripcion_problema: str = fastapi.Query(default=None, description="Descripción del problema ingresado por el ciudadano")
    peticion_solicitud: str = fastapi.Query(default=None, description="Petición del problema ingresado por el ciudadano")
    clasificacion_ciudadano: str = fastapi.Query(default=None, description="Producto - clasificación inicial ingresada por el ciudadano")
    entidad_ciudadano: str = fastapi.Query(default=None, description="Solo para los casos bancarios ingresan el banco contra el cual está reclamando")
    cantidad_documentos: int = fastapi.Query(default=None, description="Se indicará la cantidad de documentos que se enviaran como adjuntos")

class StatusClass(pydantic.BaseModel):
    Code: int = fastapi.Query(default=None, description="Códigos de estado de Respuesta HTML (¿HTTP querían decir?)")
    Value: str = fastapi.Query(default=None, description="Descripción del Código")

class RespuestaClass(pydantic.BaseModel):
    Id_caso: int = fastapi.Query(default=None, description="ID único interno del ticket de atención(enviado en la consulta)")
    Mercado: str = fastapi.Query(default=None, description="Tipo de mercado: Seguros-bancos-valores")
    Tipo_entidad: str = fastapi.Query(default=None, description="Tipo de entidad identificada ( desde el listado de entidades de la CMF)")
    Nombre_entidad: str = fastapi.Query(default=None, description="Nombre de entidad identificada ( desde el listado de entidades de la CMF)")
    Tipo_producto: str = fastapi.Query(default=None, description="Tipo producto identificados ( desde listado de tipos productos de la CMF)")
    Tipo_materia: str = fastapi.Query(default=None, description="Tipo materia identificados ( desde listado de tipos materia de la CMF)")

class ReclamoOut(pydantic.BaseModel):
    TransactionID: int = fastapi.Query(default=None, description="Código Único de la transacción")
    Status: StatusClass
    Respuesta:  RespuestaClass

class DocumentosClass(pydantic.BaseModel):
    hash_archivo: str = fastapi.Query(default=None, description="Hash registrado por el archivo")
    nombre_archivo: str = fastapi.Query(default=None, description="Nombre archivo con extensión")

class DatosCasos(pydantic.BaseModel):
    Id_caso: int = fastapi.Query(default=None, description="ID único interno del ticket de atención")
    fecha_ingreso: datetime.date = fastapi.Query(default=None, description="Fecha ingreso")
    descripcion_problema: str = fastapi.Query(default=None, description="Descripción del problema ingresado por el ciudadano")
    peticion_solicitud: str = fastapi.Query(default=None, description="Petición del problema ingresado por el ciudadano")
    clasificacion_ciudadano: str = fastapi.Query(default=None, description="Producto - clasificación inicial ingresada por el ciudadano")
    entidad_ciudadano: str = fastapi.Query(default=None, description="Solo para los casos bancarios ingresan el banco contra el cual está reclamando")
    clasificado: str = fastapi.Query(default=None, description="Señala si el reclamo ya fue clasificado por el algoritmo o no. (S/N)")
    Documentos: DocumentosClass

class DataReclamo(pydantic.BaseModel):
    fecha_consulta: datetime.date = fastapi.Query(default=None, description="Fecha consulta")
    Datos_casos: typing.List[DatosCasos] = fastapi.Query(default=None, description="Array que contiene la información para iniciar el ticket")
    
class JsonReclamos(pydantic.BaseModel):
    Data: DataReclamo

class ReclamoDiarioOut(pydantic.BaseModel):
    TransactionID: int = fastapi.Query(default=None, description="Código Único de la transacción")
    Status: StatusClass