# In this file we encode our expectations about the shapes of the well segmentations

def assert_expected_shape(i_vals: list, j_vals: list, plate_num: int) -> None:
    plate_num = int(plate_num)
    if plate_num == 24 or plate_num == 98:  # These plates have two entire missing rows off the bottom
        assert len(i_vals) == 15, f"Expected 15 i_vals for plate num {plate_num}, got {len(i_vals)}"
        assert len(j_vals) == 25, f"Expected 25 j_vals for plate num {plate_num}, got {len(j_vals)}"
    else:
        assert len(i_vals) == 17, f"Expected 17 i_vals for {plate_num}, got {len(i_vals)}"
        assert len(j_vals) == 25, f"Expected 25 j_vals for {plate_num}, got {len(j_vals)}"
