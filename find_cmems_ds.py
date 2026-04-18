import copernicusmarine

catalog = copernicusmarine.read_catalogue()

matches = []
for product in catalog.products:
    for ds in product.datasets:
        if "bgc" in ds.dataset_id.lower() and "glo" in ds.dataset_id.lower():
            matches.append(ds.dataset_id)

print(set(matches))
