import assimp_py

path = "/mnt/disk2/Stable-Renderer-Previous/Stable-Renderer/resources/example-3d-models/miku/miku.obj"

if __name__ == "__main__":
    process_flags = (assimp_py.Process_Triangulate | assimp_py.Process_CalcTangentSpace | assimp_py.Process_JoinIdenticalVertices)
    scene = assimp_py.ImportFile(str(path), process_flags)
    for mesh in scene.meshes:
        print(mesh.indices)
        print(scene.materials[mesh.material_index]['NAME'])
    print("SUCCESS!")

