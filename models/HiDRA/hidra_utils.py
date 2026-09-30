import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, concatenate, multiply, dot
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Softmax

# import numpy as np
# import pandas as pd
from tensorflow.keras.models import Model, Sequential, load_model
# from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
# from tensorflow.keras.layers import concatenate, multiply, dot, Activation
# from tensorflow.keras.utils import Sequence

def build_pathway_tensor(pathway_indices):
    max_len = max(len(p) for p in pathway_indices)
    num_pathways = len(pathway_indices)

    pathway_tensor = np.zeros((num_pathways, max_len), dtype=np.int32)
    mask = np.zeros((num_pathways, max_len), dtype=np.float32)

    for i, p in enumerate(pathway_indices):
        pathway_tensor[i, :len(p)] = p
        mask[i, :len(p)] = 1.0

    return pathway_tensor, mask

def parse_data(ic50, expr, GeneSet_Dic, drugs):
    # Divide expression data based on pathway
    X = []

    for pathway in GeneSet_Dic.keys():
        df = expr[GeneSet_Dic[pathway]]
        X.append(df.loc[ic50['improve_sample_id']])

    X.append(drugs.loc[ic50['improve_chem_id']])

    return X

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def Making_Model(num_genes, pathway_indices):

    pathway_tensor, pathway_mask = build_pathway_tensor(pathway_indices)

    pathway_tensor = tf.constant(pathway_tensor)
    pathway_mask = tf.constant(pathway_mask)

    num_pathways = pathway_tensor.shape[0]
    max_len = pathway_tensor.shape[1]

    # Inputs
    Gene_Input = Input(shape=(num_genes,), name="Gene_Input")
    Drug_Input = Input(shape=(512,), name="Drug_Input")

    # Drug network (same as before)
    d = Dense(128)(Drug_Input)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    d = Dense(32)(d)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    # new : gather genes
    # (batch, num_pathways, max_len)
    genes = tf.gather(Gene_Input, pathway_tensor, axis=1)

    # Apply mask
    genes = genes * pathway_mask

    # drug projection
    drug_proj = Dense(max_len)(d)
    drug_proj = BatchNormalization()(drug_proj)
    drug_proj = Activation('relu')(drug_proj)

    # expand to match pathways
    drug_proj = tf.expand_dims(drug_proj, axis=1)
    drug_proj = tf.tile(drug_proj, [1, num_pathways, 1])

    
    concat = tf.concat([genes, drug_proj], axis=-1)

    # new: attention
    attn = Dense(max_len, activation='tanh')(concat)
    attn = tf.nn.softmax(attn, axis=-1)

    # mask attention
    attn = attn * pathway_mask

    # dot product
    out = tf.reduce_sum(genes * attn, axis=-1)

    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # pathway level attention
    sample_attn = Dense(num_pathways, activation='tanh', name='Sample_Attention_Dense')(out)
    sample_attn = Softmax(
        axis=-1,
        name='Sample_Attention_Softmax'
    )(sample_attn)

    sample_mult = out * sample_attn

    sample_mult = BatchNormalization()(sample_mult)
    sample_mult = Activation('relu')(sample_mult)

    # final
    # drug_dense2 = Dense(128)(Drug_Input)
    # drug_dense2 = BatchNormalization()(drug_dense2)
    # drug_dense2 = Activation('relu')(drug_dense2)

    # final drug branch: original HiDRA 512 -> 256 -> 128
    drug_dense1 = Dense(256, name='Drug_Dense_1')(Drug_Input)
    drug_dense1 = BatchNormalization(name='Drug_Batch_1')(drug_dense1)
    drug_dense1 = Activation('relu', name='Drug_RELU_1')(drug_dense1)

    drug_dense2 = Dense(128, name='Drug_Dense_2')(drug_dense1)
    drug_dense2 = BatchNormalization(name='Drug_Batch_2')(drug_dense2)
    drug_dense2 = Activation('relu', name='Drug_RELU_2')(drug_dense2)

    total = tf.concat([sample_mult, drug_dense2], axis=1)

    x = Dense(128)(total)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    out = Dense(1)(x)

    model = Model(inputs=[Gene_Input, Drug_Input], outputs=out)
    return model

def gene_selection(df, genes_fpath, canc_col_name):
    GeneSet = []
    GeneSet_Dic = {}
    df_genes = set([x for x in df.columns])

    with open(genes_fpath) as f:
        for line in f:
            line = line.rstrip().split('\t')
            pathway = line[0]

            genes = [x for x in line[2:] if x in df_genes]
            GeneSet.extend(genes)
            GeneSet_Dic[pathway] = genes

    GeneSet = set(GeneSet)
    print(str(len(GeneSet)) + ' KEGG genes found in expression file.')

    genes = GeneSet.intersection(df.columns[1:])
    cols = [canc_col_name] + sorted(list(genes))

    return df[cols], GeneSet_Dic


class FastGenerator(Sequence):
    def __init__(self, expr, drug, sample_idx, drug_idx, y, batch_size, shuffle=True):
        self.expr = expr
        self.drug = drug
        self.sample_idx = sample_idx
        self.drug_idx = drug_idx
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()


    

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i*self.batch_size:(i+1)*self.batch_size]

        s_idx = self.sample_idx[idx]
        d_idx = self.drug_idx[idx]

        X_gene = self.expr[s_idx]
        X_drug = self.drug[d_idx]

        y = self.y[idx]

        return [X_gene, X_drug], y

