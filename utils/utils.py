import numpy as np

from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from utils.molecular_metrics import MolecularMetrics


def mols2grid_image(mols, molsPerRow):
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150))


def classification_report(data, model, session, sample=False):
    _, _, _, a, x, _, f, _, _ = data.next_validation_batch()

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    y_true = e.flatten()
    y_pred = a.flatten()
    target_names = [str(Chem.rdchem.BondType.values[int(e)]) for e in data.bond_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

    y_true = n.flatten()
    y_pred = x.flatten()
    target_names = [Chem.Atom(e).GetSymbol() for e in data.atom_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('\n######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))


def reconstructions(data, model, session, batch_dim=10, sample=False):
    m0, _, _, a, x, _, f, _, _ = data.next_train_batch(batch_dim)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    m1 = np.array([e if e is not None else Chem.RWMol() for e in [data.matrices2mol(n_, e_, strict=True)
                                                                  for n_, e_ in zip(n, e)]])

    mols = np.vstack((m0, m1)).T.flatten()

    return mols


def samples(data, model, session, embeddings, sample=False):
    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={
        model.embeddings: embeddings, model.training: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    return mols


def all_scores(mols, data, norm=False, reconstruction=False):
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'NP score': MolecularMetrics.natural_product_scores(mols, norm=norm),
        'QED score': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
        'logP score': MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
        'SA score': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
        'diversity score': MolecularMetrics.diversity_scores(mols, data),
        'drugcandidate score': MolecularMetrics.drugcandidate_scores(mols, data)}.items()}

    m1 = {'valid score': MolecularMetrics.valid_total_score(mols) * 100,
          'unique score': MolecularMetrics.unique_total_score(mols) * 100,
          'novel score': MolecularMetrics.novel_total_score(mols, data) * 100}

    return m0, m1
