from pt_utils.data.get_img_files import get_img_files


def test_get_image_filenames():
    test_data_dir = 'tests/test_imgs'
    filenames = get_img_files(test_data_dir)
    assert type(filenames) == list
    assert len(filenames) == 2

    filenames = get_img_files(test_data_dir, num_samples=1)
    assert len(filenames) == 1

    filenames = get_img_files(test_data_dir, num_samples=0)
    assert len(filenames) == 2
