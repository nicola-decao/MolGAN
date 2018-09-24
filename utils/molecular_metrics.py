import pickle
import gzip
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen

import math
import numpy as np

NP_model = pickle.load(gzip.open('data/NP_score.pkl.gz'))
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open('data/SA_score.pkl.gz')) for j in range(1, len(i))}


class MolecularMetrics(object):

    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda(x):
        return x is not None and Chem.MolToSmiles(x) != ''

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ''
        return x is not None and '*' not in s and '.' not in s and s != ''

    @staticmethod
    def valid_scores(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda_special, mols)), dtype=np.float32)

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean()

    @staticmethod
    def novel_scores(mols, data):
        return np.array(
            list(map(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols)))

    @staticmethod
    def novel_filter(mols, data):
        return list(filter(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols))

    @staticmethod
    def novel_total_score(mols, data):
        return MolecularMetrics.novel_scores(MolecularMetrics.valid_filter(mols), data).mean()

    @staticmethod
    def unique_scores(mols):
        smiles = list(map(lambda x: Chem.MolToSmiles(x) if MolecularMetrics.valid_lambda(x) else '', mols))
        return np.clip(
            0.75 + np.array(list(map(lambda x: 1 / smiles.count(x) if x != '' else 0, smiles)), dtype=np.float32), 0, 1)

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: Chem.MolToSmiles(x), v))
        return 0 if len(v) == 0 else len(s) / len(v)

    # @staticmethod
    # def novel_and_unique_total_score(mols, data):
    #     return ((MolecularMetrics.unique_scores(mols) == 1).astype(float) * MolecularMetrics.novel_scores(mols,
    #                                                                                                       data)).sum()
    #
    # @staticmethod
    # def reconstruction_scores(data, model, session, sample=False):
    #
    #     m0, _, _, a, x, _, f, _, _ = data.next_validation_batch()
    #     feed_dict = {model.edges_labels: a, model.nodes_labels: x, model.node_features: f, model.training: False}
    #
    #     try:
    #         feed_dict.update({model.variational: False})
    #     except AttributeError:
    #         pass
    #
    #     n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
    #         model.nodes_argmax, model.edges_argmax], feed_dict=feed_dict)
    #
    #     n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    #
    #     m1 = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
    #
    #     return np.mean([float(Chem.MolToSmiles(m0_) == Chem.MolToSmiles(m1_)) if m1_ is not None else 0
    #             for m0_, m1_ in zip(m0, m1)])

    @staticmethod
    def natural_product_scores(mols, norm=False):

        # calculating the score
        scores = [sum(NP_model.get(bit, 0)
                      for bit in Chem.rdMolDescriptors.GetMorganFingerprint(mol,
                                                                            2).GetNonzeroElements()) / float(
            mol.GetNumAtoms()) if mol is not None else None
                  for mol in mols]

        # preventing score explosion for exotic molecules
        scores = list(map(lambda score: score if score is None else (
            4 + math.log10(score - 4 + 1) if score > 4 else (
                -4 - math.log10(-4 - score + 1) if score < -4 else score)), scores))

        scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -3, 1), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        return np.array(list(map(lambda x: 0 if x is None else x, [
            MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None for mol in
            mols])))

    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        scores = [MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
                  for mol in mols]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def _compute_SAS(mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        # for bitId, v in fps.items():
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += SA_model.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - \
                 spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore

    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        scores = [MolecularMetrics._compute_SAS(mol) if mol is not None else None for mol in mols]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, 5, 1.5), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def diversity_scores(mols, data):
        rand_mols = np.random.choice(data.data, 100)
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

        scores = np.array(
            list(map(lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0, mols)))
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
        score = np.mean(dist)
        return score

    @staticmethod
    def drugcandidate_scores(mols, data):

        scores = (MolecularMetrics.constant_bump(
            MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True), 0.210,
            0.945) + MolecularMetrics.synthetic_accessibility_score_scores(mols,
                                                                           norm=True) + MolecularMetrics.novel_scores(
            mols, data) + (1 - MolecularMetrics.novel_scores(mols, data)) * 0.3) / 4

        return scores

    @staticmethod
    def constant_bump(x, x_low, x_high, decay=0.025):
        return np.select(condlist=[x <= x_low, x >= x_high],
                         choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                     np.exp(- (x - x_high) ** 2 / decay)],
                         default=np.ones_like(x))
