from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from typing import List
import logging


class BigQueryUploader:
    """
    Classe para upload e gerenciamento de dados no BigQuery
    """
    
    def __init__(self, project_id: str, dataset_id: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=project_id)
        self.logger = logging.getLogger(__name__)
    
    def create_table_schema(self) -> List[bigquery.SchemaField]:
        """
        Define schema aninhado para BigQuery
        
        VANTAGENS DO SCHEMA ANINHADO:
        - Preserva relacionamentos hierárquicos
        - Consultas SQL nativas com UNNEST
        - Armazenamento otimizado (colunar)
        - Sem necessidade de JOINs
        """
        
        schema = [
            # Campos do cabeçalho
            bigquery.SchemaField("cnpj_if", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("cnpj_gbd", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("nr_rms", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("seql_rms", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("dt_rms", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("dt_ref_base_itno", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("fmt_rms", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("cd_ocr", "STRING", mode="NULLABLE"),
            
            # Array de clientes (nome original do dataclass)
            bigquery.SchemaField(
                "Cli", "RECORD", mode="REPEATED",
                fields=[
                    bigquery.SchemaField("tip_cli", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("idfc_cli", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("nm_cli", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("ntz_rlc", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("cmdo_cli", "STRING", mode="NULLABLE"),
                    
                    # Array de operações
                    bigquery.SchemaField(
                        "operacoes", "RECORD", mode="REPEATED",
                        fields=[
                            bigquery.SchemaField("nr_unco", "STRING", mode="NULLABLE"),
                            bigquery.SchemaField("prf_ag", "STRING", mode="NULLABLE"),
                            bigquery.SchemaField("nr_ctr", "STRING", mode="NULLABLE"),
                            bigquery.SchemaField("dt_ctrc", "DATE", mode="NULLABLE"),
                            bigquery.SchemaField("cd_mdld", "STRING", mode="NULLABLE"),
                            bigquery.SchemaField("dt_aprc", "DATE", mode="NULLABLE"),
                            bigquery.SchemaField("cnpj_ctrc", "STRING", mode="NULLABLE"),
                            bigquery.SchemaField("cmdo_opr", "STRING", mode="NULLABLE"),
                            bigquery.SchemaField("idfc_cli_ctrt", "STRING", mode="NULLABLE"),
                            bigquery.SchemaField("tip_rst", "STRING", mode="NULLABLE"),
                            
                            # Detalhe da operação (nome original)
                            bigquery.SchemaField(
                                "det_opr", "RECORD", mode="NULLABLE",
                                fields=[
                                    bigquery.SchemaField("in_pre_fix", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("dt_vnct_ult_pcl", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_ctrd_fut", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("qt_pcl", "INT64", mode="NULLABLE"),
                                    bigquery.SchemaField("nr_plst_crt", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("sdo_dvdr", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("dt_vnct_opr", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("nr_gr", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("cd_ct", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("seq_ct", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("sit_ct", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_obgc", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("dt_vnct_ult_pcl_csr", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("qt_pcl_ct_csr", "INT64", mode="NULLABLE"),
                                    bigquery.SchemaField("sdo_dvdr_ct_csr", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("dt_cont_ct_csr", "DATE", mode="NULLABLE"),
                                ]
                            ),
                            
                            # Parcelas anteriores (nome original)
                            bigquery.SchemaField(
                                "pcl_ant", "RECORD", mode="REPEATED",
                                fields=[
                                    bigquery.SchemaField("dt_vnct_pcl_ant", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("perc_pcl_ant", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_pcl_ant", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("cmdo_pcl_ant", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField(
                                        "pgto_pcl_ant", "RECORD", mode="REPEATED",
                                        fields=[
                                            bigquery.SchemaField("dt_pgto_pcl_ant", "DATE", mode="NULLABLE"),
                                            bigquery.SchemaField("vl_pgto_pcl_ant", "FLOAT64", mode="NULLABLE"),
                                        ]
                                    ),
                                ]
                            ),
                            
                            # Pagamentos avulsos (nome original)
                            bigquery.SchemaField(
                                "pgto_avls", "RECORD", mode="REPEATED",
                                fields=[
                                    bigquery.SchemaField("dt_pgto_avls", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_pgto_avls", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("tip_pgto_avls", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("cmdo_pgto_avls", "STRING", mode="NULLABLE"),
                                ]
                            ),
                            
                            # Parcelas futuras (nome original)
                            bigquery.SchemaField(
                                "pcl_fut", "RECORD", mode="REPEATED",
                                fields=[
                                    bigquery.SchemaField("dt_vnct_prx_pcl", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("perc_pcl_fut", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_prx_pcl", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("qt_pcl_vncr", "INT64", mode="NULLABLE"),
                                    bigquery.SchemaField("qt_pcl_pgr", "INT64", mode="NULLABLE"),
                                ]
                            ),
                            
                            # Faturas fechadas anteriores (nome original)
                            bigquery.SchemaField(
                                "fat_fchd_ant", "RECORD", mode="REPEATED",
                                fields=[
                                    bigquery.SchemaField("dt_vnct_fat_fchd_ant", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_ttl_pgr_fat_fchd_ant", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_min_pgr_fat_fchd_ant", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("cmdo_fat_fchd_ant", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField(
                                        "pgto_fat_fchd_ant", "RECORD", mode="REPEATED",
                                        fields=[
                                            bigquery.SchemaField("dt_pgto_fat_fchd_ant", "DATE", mode="NULLABLE"),
                                            bigquery.SchemaField("vl_pgto_fat_fchd_ant", "FLOAT64", mode="NULLABLE"),
                                        ]
                                    ),
                                ]
                            ),
                            
                            # Faturas fechadas futuras (nome original)
                            bigquery.SchemaField(
                                "fat_fchd_fut", "RECORD", mode="REPEATED",
                                fields=[
                                    bigquery.SchemaField("dt_vnct_fat_fchd_fut", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_ttl_pgr_fat_fchd_fut", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_min_pgr_fat_fchd_fut", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("cmdo_fat_fch_fut", "STRING", mode="NULLABLE"),
                                ]
                            ),
                            
                            # Fechamentos anteriores (nome original)
                            bigquery.SchemaField(
                                "fcht_ant", "RECORD", mode="REPEATED",
                                fields=[
                                    bigquery.SchemaField("dt_fcht_ant", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("sdo_utlz_fcht_ant", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("cmdo_fcht_ant", "STRING", mode="NULLABLE"),
                                ]
                            ),
                            
                            # Fechamentos atuais (nome original)
                            bigquery.SchemaField(
                                "fcht_atu", "RECORD", mode="REPEATED",
                                fields=[
                                    bigquery.SchemaField("dt_fcht_atu", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("sdo_utlz_fcht_atu", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("cmdo_fcht_atu", "STRING", mode="NULLABLE"),
                                ]
                            ),
                            
                            # Parcelas anteriores consórcio (nome original)
                            bigquery.SchemaField(
                                "pcl_ant_csr", "RECORD", mode="REPEATED",
                                fields=[
                                    bigquery.SchemaField("dt_vnct_pcl_ant_csr", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("perc_pcl_ant_csr", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_pcl_ant_csr", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("cmdo_pcl_ant_csr", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField(
                                        "pgto_pcl_ant_csr", "RECORD", mode="REPEATED",
                                        fields=[
                                            bigquery.SchemaField("dt_pgto_pcl_ant_csr", "DATE", mode="NULLABLE"),
                                            bigquery.SchemaField("vl_pgto_pcl_ant_csr", "FLOAT64", mode="NULLABLE"),
                                        ]
                                    ),
                                ]
                            ),
                            
                            # Pagamentos avulsos consórcio (nome original)
                            bigquery.SchemaField(
                                "pgto_avls_csr", "RECORD", mode="REPEATED",
                                fields=[
                                    bigquery.SchemaField("dt_pgto_avls_csr", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_pgto_avls_csr", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("tip_pgto_avls_csr", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("cmdo_pgto_avls_csr", "STRING", mode="NULLABLE"),
                                ]
                            ),
                            
                            # Parcelas futuras consórcio (nome original)
                            bigquery.SchemaField(
                                "pcl_fut_csr", "RECORD", mode="REPEATED",
                                fields=[
                                    bigquery.SchemaField("dt_vnct_prx_pcl_csr", "DATE", mode="NULLABLE"),
                                    bigquery.SchemaField("perc_pcl_fut_csr", "STRING", mode="NULLABLE"),
                                    bigquery.SchemaField("vl_prx_pcl_csr", "FLOAT64", mode="NULLABLE"),
                                    bigquery.SchemaField("qt_pcl_vncr_csr", "INT64", mode="NULLABLE"),
                                    bigquery.SchemaField("qt_pcl_pgr_ct_csr", "INT64", mode="NULLABLE"),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ]
        
        return schema
    
    def create_or_update_table(self, table_id: str):
        """Cria ou atualiza tabela no BigQuery"""
        
        table_ref = self.client.dataset(self.dataset_id).table(table_id)
        
        try:
            table = self.client.get_table(table_ref)
            self.logger.info(f"Tabela {table_id} já existe")
        except NotFound:
            schema = self.create_table_schema()
            table = bigquery.Table(table_ref, schema=schema)
            
            
            #table.clustering_fields = ["Clientes"]
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="dt_rms"  
            )
            
            table = self.client.create_table(table)
            self.logger.info(f"Tabela {table_id} criada com sucesso")
        
        return table
    
    def upload_ndjson_file(self, table_id: str, ndjson_file: str):
        """Faz upload do arquivo NDJSON para BigQuery"""
        
        table_ref = self.client.dataset(self.dataset_id).table(table_id)
        
        
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
        )
        
        
        with open(ndjson_file, "rb") as source_file:
            job = self.client.load_table_from_file(
                source_file, 
                table_ref, 
                job_config=job_config
            )
        
        
        job.result()
        
        if job.errors:
            self.logger.error(f"Erro no upload: {job.errors}")
            raise RuntimeError(f"Erro no upload para BigQuery: {job.errors}")
        
        self.logger.info(f"Upload concluído: {job.output_rows} linhas carregadas")
        
        return job
