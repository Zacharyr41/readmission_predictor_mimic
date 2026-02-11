# Acknowledgements

This project builds upon foundational research in temporal knowledge graphs for clinical data analysis.

## Primary Research Foundation

This implementation is based on the conceptual modeling approach presented in:

> **A Conceptual Model for Discovering Implicit Temporal Knowledge in Clinical Data**
>
> Aurélien Vannieuwenhuyze, Nada Mimouni, Cédric Du Mouza
>
> In: Fonseca, C., Bernasconi, A., de Cesare, S., Bellatreche, L., Pastor, O. (eds) *Advances in Conceptual Modeling*. ER 2025. Lecture Notes in Computer Science, vol 16190. Springer, Cham.
>
> DOI: [10.1007/978-3-032-08620-4_6](https://doi.org/10.1007/978-3-032-08620-4_6)

The original implementation is available at: [https://github.com/avannieuwenhuyze/clinical-tkg-cmls2025](https://github.com/avannieuwenhuyze/clinical-tkg-cmls2025)

### Key Contributions from the Original Work

The foundational research introduces:

- A **competency-question-guided conceptual modeling approach** for transforming raw ICU data into temporal knowledge graphs
- An **OWL-Time-compliant ontology** for representing clinical events as temporal intervals and instants
- **Allen's interval algebra** for capturing implicit temporal relationships between clinical events
- Application to **bloodstream infection detection** through temporally grounded reasoning

Our implementation extends this work to focus on hospital readmission prediction, adding:

- Cohort selection based on ICD diagnosis codes
- Feature extraction from the temporal knowledge graph
- Machine learning model training and evaluation
- Comprehensive pipeline orchestration

## Data Source

This project uses the **MIMIC-IV** clinical database:

> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
>
> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

MIMIC-IV is restricted-access data requiring credentialed access through [PhysioNet](https://physionet.org/content/mimiciv/).

## Ontology Standards

The temporal modeling in this project follows W3C standards:

- **OWL-Time**: W3C Time Ontology in OWL. [https://www.w3.org/TR/owl-time/](https://www.w3.org/TR/owl-time/)
- **Allen's Interval Algebra**: Allen, J.F. (1983). Maintaining knowledge about temporal intervals. *Communications of the ACM*, 26(11), 832-843.

## Software Dependencies

This project relies on several open-source libraries:

- [DuckDB](https://duckdb.org/) - In-process SQL OLAP database
- [rdflib](https://rdflib.readthedocs.io/) - RDF manipulation in Python
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [NetworkX](https://networkx.org/) - Graph algorithms
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - Pre-trained language models
- [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) - Biomedical entity embeddings (Liu et al., NAACL 2021)

## Citation

If you use this work, please cite both the original research and this implementation:

```bibtex
@inproceedings{vannieuwenhuyze2025temporal,
  author    = {Vannieuwenhuyze, Aurélien and Mimouni, Nada and Du Mouza, Cédric},
  title     = {A Conceptual Model for Discovering Implicit Temporal Knowledge in Clinical Data},
  booktitle = {Advances in Conceptual Modeling},
  series    = {Lecture Notes in Computer Science},
  volume    = {16190},
  publisher = {Springer},
  year      = {2025},
  doi       = {10.1007/978-3-032-08620-4_6}
}

@software{readmission_predictor_mimic,
  title  = {Temporal Knowledge Graph-Based Hospital Readmission Prediction},
  year   = {2024},
  url    = {https://github.com/Zacharyr41/readmission_predictor_mimic}
}
```

## Contact

For questions about the original temporal knowledge graph research, contact the authors at CNAM:
- Aurélien Vannieuwenhuyze (aurelien.vannieuwenhuyze@lecnam.net)
- Nada Mimouni (nada.mimouni@lecnam.net)
- Cédric Du Mouza (cedric.dumouza@lecnam.net)
