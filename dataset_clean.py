import numpy as np
import glob
import sys
import os

read_dir = 'backup_datasets/ModelNet10/**/*.off'
path_prefix = 'backup_datasets/ModelNet10_clean_faces_removed/'
total_files_with_unused_vertices=0

def get_face_areas_and_normals(vs, faces):
    face_normals = np.cross(vs[faces[:, 1]] - vs[faces[:, 0]], vs[faces[:, 2]] - vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    return face_normals, face_areas

def add_perturbation_to_vertices(remove_faces, vs, faces):
    vs_to_perturb = []
    if remove_faces!=[]:
         for i in remove_faces:
            for j in faces[i]:
                if j not in vs_to_perturb:
                    vs_to_perturb.append(j)
         for i in vs_to_perturb:
            vs[i] = [a + mul_fact()*random.uniform(0.1, 0.2) for a in vs[i]]
         vs = np.asarray(vs)
         face_normals, face_areas = get_face_areas_and_normals(vs, faces)
         remove_faces = [ind for ind, face_area in enumerate(face_areas) if face_area == 0]
         if remove_faces!=[]:
            print("ERROR: ZERO-AREA FACES")
            sys.exit()
    return vs

def remap_vs_and_faces(list_of_unused_vertices,vs,faces):
    list_of_unused_vertices.sort(reverse=True)
    for i in range(len(list_of_unused_vertices)):
        unused_v = list_of_unused_vertices[i]
        for j in range(faces.shape[0]):
            faces[j] = [a if a < unused_v else a-1 for a in faces[j]]
        vs = np.delete(vs,unused_v,0)
    return vs, faces

def find_used_unused_vertices(vs,faces):
    list_of_used_vertices = []
    list_of_unused_vertices = []
    for i in faces:
        for j in i:
            if j not in list_of_used_vertices:
                list_of_used_vertices.append(j)
    for i in range(vs.shape[0]):
        if i not in list_of_used_vertices:
            list_of_unused_vertices.append(i)
    return list_of_used_vertices, list_of_unused_vertices

def fill_from_file(file,write_dir):
    global total_files_with_unused_vertices
    with open(file, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, __ = tuple([int(s) for s in f.readline().strip().split(' ')])
        vs = [[float(s) for s in f.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in f.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)

    if vs.shape[0] < 500 and vs.shape[0] > 200 and faces.shape[0] > 200 and faces.shape[0] < 600:
        with open('mesh_info.txt', 'a') as f:
            f.write(file+' vertices and faces '+str(vs.shape)+' '+str(faces.shape)+'\n')
        f.close()
    return

    face_normals, face_areas = get_face_areas_and_normals(vs,faces)
    remove_faces = [ind for ind, face_area in enumerate(face_areas) if face_area == 0]

    faces = [face for ind, face in enumerate(faces) if ind not in remove_faces]
    faces = np.asarray(faces, dtype=int)

    # vs = add_perturbation_to_vertices(remove_faces,vs,faces)

    list_of_used_vertices, list_of_unused_vertices = find_used_unused_vertices(vs,faces)
    # print(list_of_unused_vertices)

    vs, faces = remap_vs_and_faces(list_of_unused_vertices,vs,faces)

    if len(list_of_unused_vertices)>0:
        total_files_with_unused_vertices=total_files_with_unused_vertices+1
        # print(vs.shape)
        list_of_used_vertices, list_of_unused_vertices = find_used_unused_vertices(vs, faces)
        # print(list_of_unused_vertices)
        if list_of_unused_vertices != []:
            print("############################")
            print(file)
            print(list_of_unused_vertices)
            print("############################")
            return

    #write file
    file = path_prefix+write_dir
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        f.write('OFF\n')
        f.write(str(vs.shape[0])+' '+str(faces.shape[0])+' 0\n')
        for i in range(vs.shape[0]):
            f.write(str(vs[i][0])+' '+str(vs[i][1])+' '+str(vs[i][2])+'\n')
        for i in range(faces.shape[0]):
            f.write(str(faces[i][0])+' '+str(faces[i][1])+' '+str(faces[i][2])+'\n')
    f.close()

    # assert np.logical_and(faces >= 0, faces < len(vs)).all()
    # print(vs.shape)
    # print(faces.shape)
    # return vs, faces

def find_file_addr():
    global read_dir
    filelist=[]
    for f in glob.glob(read_dir, recursive=True):
        filelist.append(f)

    for i in range(len(filelist)):
        write_dir = filelist[i].split('ModelNet10/')[1]
        # print(filelist[i])
        print(str(i)+' of '+str(len(filelist)))
        fill_from_file(filelist[i],write_dir)
    print('total files with unused vertices: '+total_files_with_unused_vertices)

def main():
    find_file_addr()

if __name__ == "__main__":
    main()