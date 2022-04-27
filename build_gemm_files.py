import numpy as np
import glob
import sys
import os

read_dir = 'backup_datasets/ModelNet10/**/*.off'
path_prefix = 'backup_datasets/ModelNet10_clean_faces_removed/'
total_files_with_unused_vertices=0

def build_gemm(vs, faces, face_areas):
    """
    gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
    for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
    """
    ve = [[] for _ in vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                ve[edge[0]].append(edges_count)
                ve[edge[1]].append(edges_count)
                # edge_areas.append(0)
                nb_count.append(0)
                edges_count += 1
            # mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
    edges = np.array(edges, dtype=np.int32)
    gemm_edges = np.array(edge_nb, dtype=np.int64)
    sides = np.array(sides, dtype=np.int64)
    edges_count = edges_count
    # mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) / np.sum(face_areas) #todo whats the difference between edge_areas and edge_lenghts?
    file = 'debug_empty_list/chair_0382.txt'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        for i in range(len(ve)):
            f.write(str(ve[i])+'\n')
    f.close()



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

def fill_from_file(file):
    global total_files_with_unused_vertices
    with open(file, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, __ = tuple([int(s) for s in f.readline().strip().split(' ')])
        vs = [[float(s) for s in f.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in f.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    print(faces.shape)
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
    file = 'debug_empty_list/chair_0382_after_clean.off'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        f.write('OFF\n')
        f.write(str(vs.shape[0])+' '+str(faces.shape[0])+' 0\n')
        for i in range(vs.shape[0]):
            f.write(str(vs[i][0])+' '+str(vs[i][1])+' '+str(vs[i][2])+'\n')
        for i in range(faces.shape[0]):
            f.write('3 '+str(faces[i][0])+' '+str(faces[i][1])+' '+str(faces[i][2])+'\n')
    f.close()

    build_gemm(vs,faces,face_areas)


def find_file_addr():
    global read_dir

    filepath = 'ModelNet_debug/chair_0382.off'
    fill_from_file(filepath)

def main():
    find_file_addr()

if __name__ == "__main__":
    main()