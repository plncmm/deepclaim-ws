from deepclaim_ws.dummy_classifier import DummyMercadoClassifier
c = DummyMercadoClassifier()
print(c.predict(["a","b","c"]))