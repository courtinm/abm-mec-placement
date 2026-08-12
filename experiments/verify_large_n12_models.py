import os, pickle, json
base = r"output/output-urban_large_N12"
checks = []
for condition in ["fcfs", "priority"]:
    for train_seed in [0, 1, 2]:
        model_dir = os.path.join(base, condition, "models", "tseed" + str(train_seed))
        rn1 = os.path.join(model_dir, "rn_1.pkl")
        cr = os.path.join(model_dir, "cr_qtable.pkl")
        item = {"condition": condition, "train_seed": train_seed}
        item["model_dir_exists"] = os.path.isdir(model_dir)
        item["rn1_exists"] = os.path.exists(rn1)
        item["cr_exists"] = os.path.exists(cr)
        try:
            with open(rn1, "rb") as f:
                rn = pickle.load(f)
            item["rn1_loadable"] = True
            item["rn1_states"] = len(rn)
        except Exception as e:
            item["rn1_loadable"] = False
            item["rn1_error"] = type(e).__name__ + ": " + str(e)
        try:
            with open(cr, "rb") as f:
                q = pickle.load(f)
            item["cr_loadable"] = True
            item["cr_states"] = len(q)
            item["cr_actions"] = len(next(iter(q.values()))) if q else 0
        except Exception as e:
            item["cr_loadable"] = False
            item["cr_error"] = type(e).__name__ + ": " + str(e)
        checks.append(item)
print(json.dumps(checks, indent=2))
