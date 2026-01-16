# Explanation

Understanding the concepts and design decisions behind Lacuna.

## About Explanations

Explanation documentation is **understanding-oriented**. It provides context,
background, and reasoning that helps you understand *why* things work the way
they do.

## Core Concepts

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } **SubjectData**

    ---

    Understand how Lacuna manages immutable subject data containers.

    [:octicons-arrow-right-24: SubjectData Design](subject-data.md)

-   :material-head-cog:{ .lg .middle } **MaskData**

    ---

    Learn about lesion mask validation and binary requirements.

    [:octicons-arrow-right-24: MaskData Validation](mask-data.md)

-   :material-axis-arrow:{ .lg .middle } **Coordinate Spaces**

    ---

    Understand MNI spaces and spatial normalization.

    [:octicons-arrow-right-24: Coordinate Spaces](coordinate-spaces.md)

-   :material-puzzle:{ .lg .middle } **Registry Pattern**

    ---

    How Lacuna uses registries for extensibility.

    [:octicons-arrow-right-24: Registry Pattern](registries.md)

-   :material-function:{ .lg .middle } **The analyze() Function**

    ---

    How the central analyze function orchestrates workflows.

    [:octicons-arrow-right-24: Analyze Function](analyze.md)

</div>

## Why Read Explanations?

Unlike tutorials (learning) or how-to guides (doing), explanations help you:

- **Understand design decisions** — Why does Lacuna require binary masks?
- **Grasp the big picture** — How do different components fit together?
- **Make informed choices** — Which coordinate space should I use?
- **Troubleshoot effectively** — Why isn't my analysis working?

## See Also

- [Tutorials](../tutorials/index.md) — Learn Lacuna step by step
- [How-to Guides](../how-to/index.md) — Accomplish specific tasks
- [Reference](../reference/index.md) — Technical specifications
