from src.folder_multiclass import create_single_root_loaders


def create_loaders(config):
    data, training = config.section("data"), config.section("training")
    return create_single_root_loaders(
        config.path("data", "root"), int(data["image_size"]),
        int(training["batch_size"]), int(training["num_workers"]),
    )
