import random
import typing
import data


class DummyClassifier:
    def __init__(self, classes: list) -> str:
        classes = self.classes

    def predict(self, x: list):
        length = len(x)
        predictions = [random.choice(self.classes) for _ in range(length)]
        return predictions


class DummyMercadoClassifier(DummyClassifier):
    def __init__(self) -> str:
        self.classes = ["seguros", "bancos", "valores"]


class DummyTipoEntidadClassifier(DummyClassifier):
    def __init__(self) -> str:
        self.classes = ["entidad_a", "entidad_b", "entidad_c"]


class DummyNombreEntidadClassifier(DummyClassifier):
    def __init__(self) -> str:
        self.classes = ["entidad_A", "entidad_B", "entidad_C"]


class DummyTipoProductoClassifier(DummyClassifier):
    def __init__(self) -> str:
        self.classes = ["producto_a", "producto_b", "producto_c"]


class DummyTipoMateriaClassifier(DummyClassifier):
    def __init__(self) -> str:
        self.classes = ["materia_a", "materia_b", "materia_c"]


class DummyClaimClassifier(DummyClassifier):
    def __init__(self):
        self.c_mercado = DummyMercadoClassifier()
        self.c_tipo_entidad = DummyTipoEntidadClassifier()
        self.c_tipo_materia = DummyTipoMateriaClassifier()
        self.c_tipo_producto = DummyTipoProductoClassifier()
        self.c_nombre_entidad = DummyNombreEntidadClassifier()

    def predict(self, x: list) -> typing.List[data.MercadoClass]:
        mercados = []
        for s in x:
            mercado = data.MercadoClass(
                tipo_mercado=self.c_mercado.predict([s])[0],
                tipo_entidad=self.c_tipo_entidad.predict([s])[0],
                nombre_entidad=self.c_nombre_entidad.predict([s])[0],
                tipo_producto=self.c_tipo_producto.predict([s])[:2],
                tipo_materia=self.c_tipo_materia.predict([s])[:2]
            )
            mercados.append(mercado)
        return mercados
