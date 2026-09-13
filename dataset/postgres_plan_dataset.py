"""Workload-independent PostgreSQL JSON-plan input for QPPNet."""

from collections import defaultdict
import json
import random
import re

import numpy as np


NUMERIC_FIELDS = (
    "Plan Width", "Plan Rows", "Startup Cost", "Total Cost",
    "Workers Planned", "Parallel Aware",
)
CATEGORICAL_FIELDS = (
    "Relation Name", "Index Name", "Scan Direction", "Join Type",
    "Parent Relationship", "Strategy", "Partial Mode", "Sort Method",
)


def template_family(key):
    name = key.split(":", 1)[-1]
    name = re.sub(r"#\d+$", "", name)
    return re.sub(r"_s\d+$", "", name)


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            entry = json.loads(line)
            if entry.get("kind") == "plan":
                records.append(entry["record"])
    return records


def walk(node):
    yield node
    for child in node.get("Plans", []):
        yield from walk(child)


def plan_shape(node):
    return (node["Node Type"], "Subplan Name" in node,
            tuple(plan_shape(child) for child in node.get("Plans", [])))


def scale_node_times(node, scale):
    result = dict(node)
    result["__Time Scale"] = scale
    if "Plans" in node:
        result["Plans"] = [scale_node_times(child, scale)
                           for child in node["Plans"]]
    return result


class PostgresPlanDataSet:
    """QPPNet batches native PostgreSQL plans, split by template family."""

    def __init__(self, opt):
        records = load_records(opt.data_dir)
        if not records:
            raise ValueError("no PostgreSQL plans found")
        families = sorted({template_family(record["key"]) for record in records})
        random.Random(getattr(opt, "split_seed", 2027)).shuffle(families)
        split = max(1, int(len(families) * 0.8))
        train_families = set(families[:split])
        self.train_records = [record for record in records
                              if template_family(record["key"]) in train_families]
        self.test_records = [record for record in records
                             if template_family(record["key"]) not in train_families]
        if not self.test_records:
            raise ValueError("at least two template families are required")

        self.batch_size = opt.batch_size
        self.categories = {
            field: {value: number for number, value in enumerate(sorted({
                str(node[field]) for record in records
                for node in walk(record["plan"]["Plan"]) if field in node
            }))}
            for field in CATEGORICAL_FIELDS
        }
        self.local_dim = len(NUMERIC_FIELDS) + sum(
            len(values) for values in self.categories.values()
        )
        max_children = defaultdict(int)
        for record in records:
            for node in walk(record["plan"]["Plan"]):
                count = sum("Subplan Name" not in child
                            for child in node.get("Plans", []))
                max_children[node["Node Type"]] = max(
                    max_children[node["Node Type"]], count
                )
        self.dim_dict = {
            operator: self.local_dim + 32 * child_count
            for operator, child_count in max_children.items()
        }

        raw = defaultdict(list)
        for record in self.train_records:
            for node in walk(record["plan"]["Plan"]):
                raw[node["Node Type"]].append(self._raw_features(node))
        self.mean_range_dict = {
            operator: (
                np.mean(values, axis=0),
                # A feature constant in train may vary in a held-out template.
                # Unit scale avoids dividing that valid value by float epsilon.
                np.maximum(np.std(values, axis=0), 1.0),
            )
            for operator, values in raw.items()
        }
        zero = np.zeros(self.local_dim, dtype=np.float32)
        one = np.ones(self.local_dim, dtype=np.float32)
        for operator in self.dim_dict:
            self.mean_range_dict.setdefault(operator, (zero, one))

        self.dataset = self.train_records
        self.datasize = len(self.dataset)
        self.test_dataset = self._group(self.test_records)
        self.all_dataset = self._group(records)

    def _raw_features(self, node):
        numeric = [float(node.get(field, 0)) for field in NUMERIC_FIELDS]
        categorical = []
        for field, values in self.categories.items():
            encoded = [0.0] * len(values)
            if field in node:
                encoded[values[str(node[field])]] = 1.0
            categorical.extend(encoded)
        return np.asarray(numeric + categorical, dtype=np.float32)

    def _group(self, records):
        groups = defaultdict(list)
        for record in records:
            execution_ms = float(record["plan"].get(
                "Execution Time", record["label_duration_ms"]
            ))
            scale = (float(record["label_duration_ms"]) / execution_ms
                     if execution_ms > 0 else 1.0)
            root = scale_node_times(record["plan"]["Plan"], scale)
            root["__Label Duration"] = record["label_duration_ms"]
            root["__Query Key"] = record["key"]
            groups[plan_shape(root)].append(root)
        return [self.get_input(group) for group in groups.values()]

    def get_input(self, nodes):
        operator = nodes[0]["Node Type"]
        mean, scale = self.mean_range_dict[operator]
        children = [
            self.get_input([node["Plans"][index] for node in nodes])
            for index in range(len(nodes[0].get("Plans", [])))
        ]
        return {
            "node_type": operator,
            "real_node_type": operator,
            "subbatch_size": len(nodes),
            "feat_vec": np.asarray([
                (self._raw_features(node) - mean) / scale for node in nodes
            ], dtype=np.float32),
            "children_plan": children,
            "total_time": np.asarray([
                (node["__Label Duration"] if "__Label Duration" in node else
                 node["Actual Total Time"] * node["__Time Scale"]) / 100
                for node in nodes
            ], dtype=np.float32),
            "query_keys": [node["__Query Key"] for node in nodes
                           if "__Query Key" in node],
            "is_subplan": "Subplan Name" in nodes[0],
        }

    def sample_data(self):
        count = min(self.batch_size, self.datasize)
        chosen = np.random.choice(self.datasize, count, replace=False)
        return self._group([self.dataset[index] for index in chosen])
