htmls = "./docs/"
recursive = True
max_len = 200

log_files = {
    "ccapp": "./log.log",
    "cc": "../CorpusCook/log.log",
    "dist": "./ai-difference/Distinctiopus/log.log"
}

cc_corpus_collection_path = "../CorpusCook/manually_annotated/"
cc_corpus_working_path = "~/CorpusCook/server/corpus/"
dist_corpus_path = "~/ai-difference/Distinctiopus4/manual_corpus/"

mixer_path = "~/CorpusCook/manually_annotated/mix_corpus_from_manual_files.py"
mixer_working_dir =  "/roedel/home/finn/CorpusCook/"
corpuscook_venv = "/roedel/home/finn/CorpusCook/venv/bin/activate"

science_map_corpus_path="../ScienceMap/manual_corpus/"
science_map_working_dir="../ScienceMap/"
science_map="../ScienceMap/GUI.py"
science_map_venv="../ScienceMap/venv/bin/activate"
science_map_csv="../ScienceMap/manual_corpus/relations.csv"

ampligraph_working_dir="../allennlp_vs_ampligraph/"
ampligraph_venv="../allennlp_vs_ampligraph/venv/bin/activate"
ampligraph="../allennlp_vs_ampligraph/csv_ampligraph.py"
ampligraph_coords="CONSTRASTSUBJECT"

train_venv_python = "/roedel/home/finn/ai-difference/venv/bin/activate"
train_path= "/roedel/home/finn/ai-difference/Distinctiopus4"
train_script= "/roedel/home/finn//ai-difference/Distinctiopus4/do/train_multi_corpus.py"
train_log = "train.log"
allennlp_config = "/roedel/home/finn//ai-difference/Distinctiopus4/experiment_configs/elmo_lstm3_feedforward4_crf_straight_fitter.config"

dist_model_path_first = "/roedel/home/finn/ai-difference/Distinctiopus4/output/first_./experiment_configs/{config}/model.tar.gz".format(config=allennlp_config)
cc_model_path_first   = "/roedel/home/finn/CorpusCook/server/models/model_first.tar.gz"
dist_model_path_over  = "/roedel/home/finn/ai-difference/Distinctiopus4/output/over_./experiment_configs/{config}/model.tar.gz".format(config=allennlp_config)
cc_model_path_over    = "/roedel/home/finn/CorpusCook/server/models/model_over.tar.gz"

all_coordinates="/home/stefan/PycharmProjects/allennlp_vs_ampligraph/knowledge_graph_coords/knowledge_graph_3d_choords.csv"
ke_path=  "/home/stefan/PycharmProjects/allennlp_vs_ampligraph/knowledge_graph_coords/tsne_clusters_mean_points.csv"
ke_colors="/home/stefan/PycharmProjects/allennlp_vs_ampligraph/knowledge_graph_coords/kn_clusters_mean_points.csv"
hal = '"/home/stefan/IdeaProjects/hal/target/hal-1-jar-with-dependencies.jar"'
video_dir = '/home/stefan/IdeaProjects/view_control_web/WebContent/resources/media/'
