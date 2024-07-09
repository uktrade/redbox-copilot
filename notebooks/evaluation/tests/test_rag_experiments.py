import json
import sys
from pathlib import Path

# Adding the appropriate directory to sys.path
sys.path.append(str(Path(__file__).parents[2]))
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (AnswerRelevancyMetric, ContextualPrecisionMetric,
                              ContextualRecallMetric,
                              ContextualRelevancyMetric, FaithfulnessMetric,
                              HallucinationMetric)

from evaluation.rag_experiments import GetExperimentResults


class TestGetExperimentResults(unittest.TestCase):
    
    def setUp(self):
        self.get_experiment_results = GetExperimentResults()
        self.get_experiment_results.set_data_version('0.2.3')
        
        # Mocking the environment setup
        self.get_experiment_results.ENV = MagicMock()
        self.get_experiment_results.ENV.elasticsearch_client.return_value = MagicMock()
        self.get_experiment_results.ENV.embedding_model = "all-mpnet-base-v2"
        
        self.get_experiment_results.ES_CLIENT = self.get_experiment_results.ENV.elasticsearch_client()
        
        # Mock paths
        self.get_experiment_results.V_EMBEDDINGS = Path('/mock/embeddings')
        self.get_experiment_results.V_SYNTHETIC = Path('/mock/synthetic')
        self.get_experiment_results.V_RESULTS = Path('/mock/results')

    @patch('jsonlines.open')
    def test_load_chunks_from_jsonl_to_index(self, mock_jsonlines_open):
        mock_jsonlines_open.return_value.__enter__.return_value = iter([json.dumps({
            "uuid": "1234",
            "parent_file_uuid": "abcd",
            "data": "test data"
        })])
        
        file_uuids = self.get_experiment_results.load_chunks_from_jsonl_to_index()
        self.assertIn("abcd", file_uuids)
        self.get_experiment_results.ES_CLIENT.index.assert_called_once()
        
    @patch('elasticsearh.helpers.bulk')
    @patch('elasticsearh.helpers.scan')
    def test_clear_index(self, mock_scan, mock_bulk):
        mock_scan.return_value = iter([{'_index': 'test_index', '_id': 'test_id'}])
        
        self.get_experiment_results.clear_index()
        self.get_experiment_results.ES_CLIENT.indices.exists.assert_called_once()
        mock_bulk.assert_called_once()

    def test_load_experiment_param_data(self):
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                'experiment_name': ['test_experiment'],
                'retrieval_system_prompt': ['test_prompt'],
                'retrieval_question_prompt': ['test_question_prompt']
            })
            
            self.get_experiment_results.load_experiment_param_data('test_file')
            self.assertEqual(self.get_experiment_results.experiment_name, 'test_experiment')
            
    @patch('evaluation.rag_experiments.GetExperimentResults.get_rag_results')
    @patch('pandas.DataFrame.to_csv')
    def test_write_rag_results(self, mock_to_csv, mock_get_rag_results):
        mock_get_rag_results.return_value = {
            'output_text': 'test_output',
            'source_documents': [{'page_content': 'test_content'}]
        }
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                'input': ['test_input']
            })
            self.get_experiment_results.write_rag_results()
            mock_to_csv.assert_called_once()

    @patch('evaluation.rag_experiments.evaluate')
    @patch('pandas.DataFrame.to_csv')
    def test_do_evaluation(self, mock_to_csv, mock_evaluate):
        mock_evaluate.return_value = []
        
        self.get_experiment_results.do_evaluation()
        mock_to_csv.assert_called_once()

    @patch('pandas.DataFrame.to_csv')
    def test_write_evaluation_results(self, mock_to_csv):
        self.get_experiment_results.eval_results = []
        
        self.get_experiment_results.write_evaluation_results()
        mock_to_csv.assert_called_once()

    @patch('seaborn.barplot')
    @patch('pandas.concat')
    @patch('pandas.read_csv')
    def test_create_visualisation_plus_grouped_results(self, mock_read_csv, mock_concat, mock_barplot):
        mock_read_csv.return_value = pd.DataFrame({
            'experiment_name': ['test_experiment'],
            'score': [0.5],
            'metric_name': ['test_metric']
        })
        mock_concat.return_value = mock_read_csv.return_value
        
        self.get_experiment_results.create_visualisation_plus_grouped_results()
        mock_barplot.assert_called_once()

if __name__ == '__main__':
    unittest.main()
