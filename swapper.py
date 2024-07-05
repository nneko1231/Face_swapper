import os
import cv2
import copy
import argparse
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple

# Mengambil model face swap dari path yang diberikan
def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model

# Menginisialisasi dan mengembalikan objek FaceAnalysis
def getFaceAnalyser(model_path: str, providers, det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser

# Mengambil satu wajah dari frame yang diberikan menggunakan face_analyser
def get_one_face(face_analyser, frame: np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

# Mengambil banyak wajah dari frame yang diberikan menggunakan face_analyser
def get_many_faces(face_analyser, frame: np.ndarray):
    """
    Mengambil wajah dari kiri ke kanan secara berurutan
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

# Menukar wajah dari source_faces ke target_faces dalam frame sementara
def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    """
    Menempelkan source_face pada target image
    """
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]

    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

# Memproses penggantian wajah antara source_img dan target_img menggunakan model yang diberikan
def process(source_img: Union[Image.Image, List], target_img: Image.Image, source_indexes: str, target_indexes: str, model: str):
    # Memuat providers yang tersedia pada mesin
    providers = onnxruntime.get_available_providers()

    # Memuat face_analyser
    face_analyser = getFaceAnalyser(model, providers)
    
    # Memuat face_swapper
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = getFaceSwapModel(model_path)
    
    # Membaca target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    # Mendeteksi wajah yang akan diganti pada target image
    target_faces = get_many_faces(face_analyser, target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if isinstance(source_img, list) and num_source_images == num_target_faces:
            print("Mengganti wajah pada target image dari kiri ke kanan secara berurutan")
            for i in range(num_target_faces):
                source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                source_index = i
                target_index = i

                if source_faces is None:
                    raise Exception("Tidak ditemukan wajah pada source!")

                temp_frame = swap_face(
                    face_swapper,
                    source_faces,
                    target_faces,
                    source_index,
                    target_index,
                    temp_frame
                )
        elif num_source_images == 1:
            # Mendeteksi wajah pada source image yang akan diganti ke target image
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            num_source_faces = len(source_faces)
            print(f"Wajah pada source: {num_source_faces}")
            print(f"Wajah pada target: {num_target_faces}")

            if source_faces is None:
                raise Exception("Tidak ditemukan wajah pada source!")

            if target_indexes == "-1":
                if num_source_faces == 1:
                    print("Mengganti semua wajah pada target image dengan satu wajah pada source image")
                    num_iterations = num_target_faces
                elif num_source_faces < num_target_faces:
                    print("Jumlah wajah pada source lebih sedikit daripada target, mengganti sebanyak yang bisa")
                    num_iterations = num_source_faces
                elif num_target_faces < num_source_faces:
                    print("Jumlah wajah pada target lebih sedikit daripada source, mengganti sebanyak yang bisa")
                    num_iterations = num_target_faces
                else:
                    print("Mengganti semua wajah pada target image dengan wajah pada source image")
                    num_iterations = num_target_faces

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    target_index = i

                    temp_frame = swap_face(
                        face_swapper,
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame
                    )
            else:
                print("Mengganti wajah tertentu pada target image dengan wajah tertentu pada source image")

                if source_indexes == "-1":
                    source_indexes = ','.join(map(lambda x: str(x), range(num_source_faces)))

                if target_indexes == "-1":
                    target_indexes = ','.join(map(lambda x: str(x), range(num_target_faces)))

                source_indexes = source_indexes.split(',')
                target_indexes = target_indexes.split(',')
                num_source_faces_to_swap = len(source_indexes)
                num_target_faces_to_swap = len(target_indexes)

                if num_source_faces_to_swap > num_source_faces:
                    raise Exception("Jumlah indeks wajah pada source lebih banyak daripada jumlah wajah pada source image")

                if num_target_faces_to_swap > num_target_faces:
                    raise Exception("Jumlah indeks wajah pada target lebih banyak daripada jumlah wajah pada target image")

                if num_source_faces_to_swap > num_target_faces_to_swap:
                    num_iterations = num_source_faces_to_swap
                else:
                    num_iterations = num_target_faces_to_swap

                if num_source_faces_to_swap == num_target_faces_to_swap:
                    for index in range(num_iterations):
                        source_index = int(source_indexes[index])
                        target_index = int(target_indexes[index])

                        if source_index > num_source_faces-1:
                            raise ValueError(f"Indeks wajah pada source {source_index} lebih tinggi daripada jumlah wajah pada source image")

                        if target_index > num_target_faces-1:
                            raise ValueError(f"Indeks wajah pada target {target_index} lebih tinggi daripada jumlah wajah pada target image")

                        temp_frame = swap_face(
                            face_swapper,
                            source_faces,
                            target_faces,
                            source_index,
                            target_index,
                            temp_frame
                        )
        else:
            raise Exception("Konfigurasi wajah tidak didukung")
        result = temp_frame
    else:
        print("Tidak ditemukan wajah pada target!")
    
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image

# Parsing argument dari command line
def parse_args():
    parser = argparse.ArgumentParser(description="Face swap.")
    parser.add_argument("--source_img", type=str, required=True, help="Path dari source image, bisa berupa beberapa gambar, dir;dir2;dir3.")
    parser.add_argument("--target_img", type=str, required=True, help="Path dari target image.")
    parser.add_argument("--output_img", type=str, required=False, default="result.png", help="Path dan nama file dari output image.")
    parser.add_argument("--source_indexes", type=str, required=False, default="-1", help="Daftar indeks wajah yang dipisahkan oleh koma untuk digunakan pada source image, dimulai dari 0 (-1 menggunakan semua wajah pada source image)")
    parser.add_argument("--target_indexes", type=str, required=False, default="-1", help="Daftar indeks wajah yang dipisahkan oleh koma untuk diganti pada target image, dimulai dari 0 (-1 mengganti semua wajah pada target image)")
    parser.add_argument("--face_restore", action="store_true", help="Flag untuk restorasi wajah.")
    parser.add_argument("--background_enhance", action="store_true", help="Flag untuk peningkatan background.")
    parser.add_argument("--face_upsample", action="store_true", help="Flag untuk upsample wajah.")
    parser.add_argument("--upscale", type=int, default=1, help="Nilai upscaling, hingga 4.")
    parser.add_argument("--codeformer_fidelity", type=float, default=0.5, help="Fidelity codeformer.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    source_img_paths = args.source_img.split(';')
    print("Path source image:", source_img_paths)
    target_img_path = args.target_img
    
    source_img = [Image.open(img_path) for img_path in source_img_paths]
    target_img = Image.open(target_img_path)

    # Mengunduh dari https://huggingface.co/deepinsight/inswapper/tree/main
    model = "./checkpoints/inswapper_128.onnx"
    result_image = process(source_img, target_img, args.source_indexes, args.target_indexes, model)
    
    if args.face_restore:
        from restoration import *
        
        # Memastikan ckpts terunduh dengan sukses
        check_ckpts()
        
        # https://huggingface.co/spaces/sczhou/CodeFormer
        upsampler = set_realesrgan()
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"],).to(device)
        ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path)["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        codeformer_net.eval()
        
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = face_restoration(result_image, args.background_enhance, args.face_upsample, args.upscale, args.codeformer_fidelity, upsampler, codeformer_net, device)
        result_image = Image.fromarray(result_image)
    
    # Menyimpan hasil
    result_image.save(args.output_img)
    print(f'Hasil berhasil disimpan: {args.output_img}')
