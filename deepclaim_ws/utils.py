def load_model(model, tokenizer, dir, device):
    print("Cargando el modelo ...")
    model = model.from_pretrained(dir)
    tokenizer = tokenizer.from_pretrained(dir)
    model.to(device)
    return model