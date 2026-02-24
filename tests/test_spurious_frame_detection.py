import unittest

from chlamy_impi.error_correction.spurious_frame_detection import detect_spurious_frames
from chlamy_impi.paths import get_npy_and_csv_filenames_given_basename


class TestSpuriousFrameDetection(unittest.TestCase):
    def test_spurious_frame_detection_known_data(self):
        filename_to_erroneous_frames = get_filename_to_erroneous_frames()

        correct = 0

        for basename in filename_to_erroneous_frames:
            df, img_arr = get_npy_and_csv_filenames_given_basename(basename)
            spurious_indices, spurious_frames = detect_spurious_frames(df, img_arr, basename)

            pred_errors = list(zip(spurious_indices, spurious_frames))
            gt_errors = filename_to_erroneous_frames[basename]
            #self.assertEqual(pred_errors, gt_errors, f"Error for {basename}, expected {gt_errors}, got {pred_errors}")

            if pred_errors == gt_errors:
                correct += 1

        print(f"Correct: {correct}/{len(filename_to_erroneous_frames)}")





def get_filename_to_erroneous_frames() -> dict[str, tuple[int, int]]:
    """In this function I have manually recorded the (meta_df_index, frame index) pairs for files which exhibit
    the known camera error in which a single erroneous frame is inserted into the image stack. A new line also appears
    in the corresponding .csv file, which we need to remove.
    """
    filename_to_inds = {
        '20231206_99-M6_30s-30s': [(61, 122)],
        '20240223_16-M6_30s-30s': [(23, 46)],
        '20240330_23-M6_30s-30s': [(61, 122)],
        '20240418_9-M6_30s-30s': [(47, 94)],
        '20240422_6-M6_30s-30s': [(9, 18)],
        '20240424_12-M6_30s-30s': [(11, 22), (26, 51)],
        '20240502_17-M6_30s-30s': [(15, 30), (74, 147)],
        '20231024_3-M1_1min-1min': [(77, 154)],
        '20231031_4-M2_1min-1min': [(57, 114)],
        '20231105_5-M1_1min-1min': [(37, 74), (38, 75)],
        '20231117_7-M1_1min-1min': [(61, 122)],
        '20240313_21-M1_1min-1min': [(13, 26), (24, 47)],
        '20240301_18-M1_1min-1min': [(11, 22)],
        '20240308_17-M2_1min-1min': [(35, 70)],
        '20240406_20-M1_1min-1min': [(71, 142)],
        '20240506_1-M5_10min-10min': [(31, 62)],
        '20240130_12-M6_10min-10min': [(35, 70)],
        '20240305_18-M5_10min-10min': [(67, 134)],
        '20240323_22-M5_10min-10min': [(69, 138)],
        '20240412_2-M5_10min-10min': [(60, 120)],
        #'20240405_1-M6_10min-10min': [(42, 84), (71, 141)],  # This filename has changed on the drive since the manual error finding was done
        '20240215_15-M4_2h-2h': [(18, 36)],
        '20231027_3-M4_2h-2h': [(40, 80)],
        '20240316_21-M4_2h-2h': [(17, 34)],
        '20240429_24-M4_2h-2h': [(6, 12)],
        '20231217_10-M3_20h_ML': [(21, 42)],
        '20240220_16-M3_20h_ML': [(12, 24)],
        '20240327_23-M3_20h_ML': [(4, 8)],
        '20240503_21-M3_20h_ML': [(36, 72)],
        '20231025_3-M2_20h_HL': [(15, 30)],
        '20231112_6-M2_20h_HL': [(34, 68)],
        '20240225_8-M2_20h_HL': [(36, 72)],
        '20240427_24-M2_20h_HL': [(4, 8)],
    }

    return filename_to_inds
