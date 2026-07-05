# face_module/cluster.py
"""人脸聚类逻辑 — 余弦相似度 + 连通分量分组。

对外由 cluster_faces() 包装调用，A 不直接使用 FaceClusterer。
"""

from collections import defaultdict

from .schema import FaceClusterResult, PersonCluster


class FaceClusterer:
    """人脸聚类器。

    v1 算法：余弦相似度 + 连通分量。
    v2 预留 method="dbscan" 参数。
    """

    def cluster(
        self,
        faces: list,
        method: str = "cosine_threshold",
        threshold: float = 0.5,
    ) -> FaceClusterResult:
        """对所有检测到的人脸执行聚类。

        Args:
            faces: list[FaceResult]，每个必须含 embedding
            method: "cosine_threshold"（v1）或 "dbscan"（v2 预留）
            threshold: 余弦相似度阈值，默认 0.5

        Returns:
            FaceClusterResult：包含 persons、已回填 person_id 的 faces、asset_tags 汇总

        Raises:
            ValueError: method 不支持或 faces 为空
        """
        if not faces:
            raise ValueError("faces 不能为空，请先调用 analyze_faces()")

        if method == "cosine_threshold":
            return self._cluster_by_threshold(faces, threshold)
        elif method == "dbscan":
            raise NotImplementedError("DBSCAN 聚类将在 v2 支持")
        else:
            raise ValueError(
                f"不支持的聚类方法: {method}，可选: cosine_threshold, dbscan"
            )

    def _cluster_by_threshold(
        self, faces: list, threshold: float
    ) -> FaceClusterResult:
        """余弦相似度阈值聚类 + 连通分量分组。"""
        import numpy as np

        # 提取 embedding 矩阵 (N, D)
        embeddings = np.array([f.embedding for f in faces], dtype=np.float32)
        n = len(embeddings)

        # L2 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        # 余弦相似度矩阵
        sim_matrix = embeddings @ embeddings.T

        # 并查集连通分量
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > threshold:
                    union(i, j)

        # 收集分组
        groups = defaultdict(list)
        for idx in range(n):
            groups[find(idx)].append(idx)

        # 按组大小降序分配 person_id
        sorted_groups = sorted(groups.values(), key=len, reverse=True)

        persons = []
        asset_faces_map = defaultdict(list)  # asset_id -> list[person_id]

        for person_idx, group in enumerate(sorted_groups, start=1):
            person_id = f"person_{person_idx:04d}"

            # 选代表照：组内 quality 最高
            best_face = max(group, key=lambda idx: faces[idx].quality)
            prototype_face_id = faces[best_face].face_id

            # 收集该人物的 asset_ids
            asset_ids = []
            for idx in group:
                asset_id = faces[idx].asset_id
                asset_faces_map[asset_id].append(person_id)
                if asset_id not in asset_ids:
                    asset_ids.append(asset_id)

            persons.append(PersonCluster(
                person_id=person_id,
                asset_ids=asset_ids,
                face_count=len(group),
                prototype_face_id=prototype_face_id,
            ))

            # 回填 person_id 到每个 FaceResult
            for idx in group:
                faces[idx].person_id = person_id

        # 生成 asset_tags 汇总
        asset_tags = []
        for asset_id, person_ids in asset_faces_map.items():
            tags = ["has_face"] + sorted(set(person_ids))
            asset_tags.append({
                "asset_id": asset_id,
                "summary_tags": tags,
            })

        return FaceClusterResult(
            persons=persons,
            faces=faces,
            asset_tags=asset_tags,
        )
