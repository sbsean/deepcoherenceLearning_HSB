import os, glob


def load_candidate_paths_from_folders(base_dir, folder_list):
    candidate_paths = {}
    for folder in folder_list:
        folder_path = os.path.join(base_dir, folder)
        pt_files = sorted(glob.glob(os.path.join(folder_path, "*.pt")))  # 정렬 보장
        idx_counter = 0  # 모든 폴더에서 사용할 인덱스 초기화

        for pt_file in pt_files:
            if pt_file.endswith("result_dict.pt"):
                continue

            basename = os.path.splitext(os.path.basename(pt_file))[0]
            if not basename.startswith("one_angle_tensor_"):
                continue

            if folder == "PICMUS":
                key = (folder, idx_counter)
                idx_counter += 1
            else:
                try:
                    acq = int(basename[len("one_angle_tensor_") :])
                    key = (folder, acq)
                except Exception:
                    continue

            candidate_paths[key] = pt_file

    return candidate_paths
