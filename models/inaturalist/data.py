from src.folder_multiclass import create_external_test_loaders


def create_loaders(config):
    data, training = config.section("data"), config.section("training")
    return create_external_test_loaders(
        config.path("data", "train_root"), config.path("data", "test_root"),
        int(data["image_size"]), int(training["batch_size"]),
        int(training["num_workers"]),
    )
