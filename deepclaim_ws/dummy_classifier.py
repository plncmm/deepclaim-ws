import random

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