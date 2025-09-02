import xml.etree.ElementTree as ET
from data_class import *
from typing import Optional
import os

class HistoricoCreditoXMLParser:
    """Parser completo para XML de histórico de crédito"""
    
    def __init__(self):
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    
    def _get_text(self, element: ET.Element, tag: str) -> Optional[str]:
        """Extrai texto de um elemento XML"""
        if element is None:
            return None
        child = element.find(tag)
        return child.text.strip() if child is not None and child.text else None
    
    def _get_float(self, element: ET.Element, tag: str) -> Optional[float]:
        """Extrai valor float de um elemento XML"""
        text = self._get_text(element, tag)
        try:
            return float(text) if text else None
        except (ValueError, TypeError):
            self.logger.warning(f"Erro ao converter '{text}' para float no campo '{tag}'")
            return None
    
    def _get_int(self, element: ET.Element, tag: str) -> Optional[int]:
        """Extrai valor inteiro de um elemento XML"""
        text = self._get_text(element, tag)
        try:
            return int(text) if text else None
        except (ValueError, TypeError):
            self.logger.warning(f"Erro ao converter '{text}' para int no campo '{tag}'")
            return None
    
    
    def parse_pagamento_parcela_anterior(self, element: ET.Element) -> PagamentoParcelaAnterior:
        """Parser para PagamentoParcelaAnterior"""
        return PagamentoParcelaAnterior(
            dt_pgto_pcl_ant=self._get_text(element, 'DtPgtoPclAnt'),
            vl_pgto_pcl_ant=self._get_float(element, 'VlPgtoPclAnt')
        )
    
    def parse_parcela_anterior(self, element: ET.Element) -> ParcelaAnterior:
        """Parser para ParcelaAnterior"""
        parcela = ParcelaAnterior(
            dt_vnct_pcl_ant=self._get_text(element, 'DtVnctPclAnt'),
            perc_pcl_ant=self._get_text(element, 'PercPclAnt'),
            vl_pcl_ant=self._get_float(element, 'VlPclAnt'),
            cmdo_pcl_ant=self._get_text(element, 'CmdoPclAnt')
        )
        
        for pgto_elem in element.findall('PgtoPclAnt'):
            pagamento = self.parse_pagamento_parcela_anterior(pgto_elem)
            parcela.pgto_pcl_ant.append(pagamento)
        
        return parcela
    def parse_pagamento_avulso(self, element: ET.Element) -> PagamentoAvulso:
        """Parser para PagamentoAvulso"""
        return PagamentoAvulso(
            dt_pgto_avls=self._get_text(element, 'DtPgtoAvls'),
            vl_pgto_avls=self._get_float(element, 'VlPgtoAvls'),
            tip_pgto_avls=self._get_text(element, 'TipPgtoAvls'),
            cmdo_pgto_avls=self._get_text(element, 'CmdoPgtoAvls')
        )
    
    def parse_parcelas_futuras(self, element: ET.Element) -> ParcelasFuturas:
        """Parser para ParcelasFuturas"""
        return ParcelasFuturas(
            dt_vnct_prx_pcl=self._get_text(element, 'DtVnctPrxPcl'),
            perc_pcl_fut=self._get_text(element, 'PercPclFut'),
            vl_prx_pcl=self._get_float(element, 'VlPrxPcl'),
            qt_pcl_vncr=self._get_int(element, 'QtPclVncr'),
            qt_pcl_pgr=self._get_int(element, 'QtPclPgr')
        )
    
    def parse_pagamento_fatura_fechada_anterior(self, element: ET.Element) -> PagamentoFaturaFechadaAnterior:
        """Parser para PagamentoFaturaFechadaAnterior"""
        return PagamentoFaturaFechadaAnterior(
            dt_pgto_fat_fchd_ant=self._get_text(element, 'DtPgtoFatFchdAnt'),
            vl_pgto_fat_fchd_ant=self._get_float(element, 'VlPgtoFatFchdAnt')
        )
    
    def parse_fatura_fechada_anterior(self, element: ET.Element) -> FaturaFechadaAnterior:
        """Parser para FaturaFechadaAnterior"""
        fatura = FaturaFechadaAnterior(
            dt_vnct_fat_fchd_ant=self._get_text(element, 'DtVnctFatFchdAnt'),
            vl_ttl_pgr_fat_fchd_ant=self._get_float(element, 'VlTtlPgrFatFchdAnt'),
            vl_min_pgr_fat_fchd_ant=self._get_float(element, 'VlMinPgrFatFchdAnt'),
            cmdo_fat_fchd_ant=self._get_text(element, 'CmdoFatFchdAnt')
        )
        
        for pgto_elem in element.findall('PgtoFatFchdAnt'):
            pagamento = self.parse_pagamento_fatura_fechada_anterior(pgto_elem)
            fatura.pgto_fat_fchd_ant.append(pagamento)
        
        return fatura
    
    def parse_fatura_fechada_futura(self, element: ET.Element) -> FaturaFechadaFutura:
        """Parser para FaturaFechadaFutura"""
        return FaturaFechadaFutura(
            dt_vnct_fat_fchd_fut=self._get_text(element, 'DtVnctFatFchdFut'),
            vl_ttl_pgr_fat_fchd_fut=self._get_float(element, 'VlTtlPgrFatFchdFut'),
            vl_min_pgr_fat_fchd_fut=self._get_float(element, 'VlMinPgrFatFchdFut'),
            cmdo_fat_fch_fut=self._get_text(element, 'CmdoFatFchFut')
        )
    
    def parse_fechamento_anterior(self, element: ET.Element) -> FechamentoAnterior:
        """Parser para FechamentoAnterior"""
        return FechamentoAnterior(
            dt_fcht_ant=self._get_text(element, 'DtFchtAnt'),
            sdo_utlz_fcht_ant=self._get_float(element, 'SdoUtlzFchtAnt'),
            cmdo_fcht_ant=self._get_text(element, 'CmdoFchtAnt')
        )
    
    def parse_fechamento_atual(self, element: ET.Element) -> FechamentoAtual:
        """Parser para FechamentoAtual"""
        return FechamentoAtual(
            dt_fcht_atu=self._get_text(element, 'DtFchtAtu'),
            sdo_utlz_fcht_atu=self._get_float(element, 'SdoUtlzFchtAtu'),
            cmdo_fcht_atu=self._get_text(element, 'CmdoFchtAtu')
        )
    
    def parse_pagamento_parcela_anterior_consorcio(self, element: ET.Element) -> PagamentoParcelaAnteriorConsorcio:
        """Parser para PagamentoParcelaAnteriorConsorcio"""
        return PagamentoParcelaAnteriorConsorcio(
            dt_pgto_pcl_ant_csr=self._get_text(element, 'DtPgtoPclAntCsr'),
            vl_pgto_pcl_ant_csr=self._get_float(element, 'VlPgtoPclAntCsr')
        )
    
    def parse_parcela_anterior_consorcio(self, element: ET.Element) -> ParcelaAnteriorConsorcio:
        """Parser para ParcelaAnteriorConsorcio"""
        parcela = ParcelaAnteriorConsorcio(
            dt_vnct_pcl_ant_csr=self._get_text(element, 'DtVnctPclAntCsr'),
            perc_pcl_ant_csr=self._get_text(element, 'PercPclAntCsr'),
            vl_pcl_ant_csr=self._get_float(element, 'VlPclAntCsr'),
            cmdo_pcl_ant_csr=self._get_text(element, 'CmdoPclAntCsr')
        )
        
        for pgto_elem in element.findall('PgtoPclAntCsr'):
            pagamento = self.parse_pagamento_parcela_anterior_consorcio(pgto_elem)
            parcela.pgto_pcl_ant_csr.append(pagamento)
        
        return parcela
    
    def parse_pagamento_avulso_consorcio(self, element: ET.Element) -> PagamentoAvulsoConsorcio:
        """Parser para PagamentoAvulsoConsorcio"""
        return PagamentoAvulsoConsorcio(
            dt_pgto_avls_csr=self._get_text(element, 'DtPgtoAvlsCsr'),
            vl_pgto_avls_csr=self._get_float(element, 'VlPgtoAvlsCsr'),
            tip_pgto_avls_csr=self._get_text(element, 'TipPgtoAvlsCsr'),
            cmdo_pgto_avls_csr=self._get_text(element, 'CmdoPgtoAvlsCsr')
        )
    
    def parse_parcelas_futuras_consorcio(self, element: ET.Element) -> ParcelasFuturasConsorcio:
        """Parser para ParcelasFuturasConsorcio"""
        return ParcelasFuturasConsorcio(
            dt_vnct_prx_pcl_csr=self._get_text(element, 'DtVnctPrxPclCsr'),
            perc_pcl_fut_csr=self._get_text(element, 'PercPclFutCsr'),
            vl_prx_pcl_csr=self._get_float(element, 'VlPrxPclCsr'),
            qt_pcl_vncr_csr=self._get_int(element, 'QtPclVncrCsr'),
            qt_pcl_pgr_ct_csr=self._get_int(element, 'QtPclPgrCtCsr')
        )
    
    def parse_detalhe_operacao(self, element: ET.Element) -> DetalheOperacao:
        """Parser para DetalheOperacao"""
        return DetalheOperacao(
            in_pre_fix=self._get_text(element, 'InPreFix'),
            dt_vnct_ult_pcl=self._get_text(element, 'DtVnctUltPcl'),
            vl_ctrd_fut=self._get_float(element, 'VlCtrdFut'),
            qt_pcl=self._get_int(element, 'QtPcl'),
            nr_plst_crt=self._get_text(element, 'NrPlstCrt'),
            sdo_dvdr=self._get_float(element, 'SdoDvdr'),
            dt_vnct_opr=self._get_text(element, 'DtVnctOpr'),
            nr_gr=self._get_text(element, 'NrGr'),
            cd_ct=self._get_text(element, 'CdCt'),
            seq_ct=self._get_text(element, 'SeqCt'),
            sit_ct=self._get_text(element, 'SitCt'),
            vl_obgc=self._get_float(element, 'VlObgc'),
            dt_vnct_ult_pcl_csr=self._get_text(element, 'DtVnctUltPclCsr'),
            qt_pcl_ct_csr=self._get_int(element, 'QtPclCtCsr'),
            sdo_dvdr_ct_csr=self._get_float(element, 'SdoDvdrCtCsr'),
            dt_cont_ct_csr=self._get_text(element, 'DtContCtCsr')
        )
    
    def parse_operacao(self, element: ET.Element) -> Operacao:
        """Parser para Operacao"""
        operacao = Operacao(
            nr_unco=self._get_text(element, 'NrUnco'),
            prf_ag=self._get_text(element, 'PrfAg'),
            nr_ctr=self._get_text(element, 'NrCtr'),
            dt_ctrc=self._get_text(element, 'DtCtrc'),
            cd_mdld=self._get_text(element, 'CdMdld'),
            dt_aprc=self._get_text(element, 'DtAprc'),
            cnpj_ctrc=self._get_text(element, 'CnpjCtrc'),
            cmdo_opr=self._get_text(element, 'CmdoOpr'),
            idfc_cli_ctrt=self._get_text(element, 'IdfcCliCtrt'),
            tip_rst=self._get_text(element, 'TipRst')
        )
        
        det_opr_elem = element.find('DetOpr')
        if det_opr_elem is not None:
            operacao.det_opr = self.parse_detalhe_operacao(det_opr_elem)
        
        for pcl_elem in element.findall('PclAnt'):
            parcela = self.parse_parcela_anterior(pcl_elem)
            operacao.pcl_ant.append(parcela)
        
        for pgto_elem in element.findall('PgtoAvls'):
            pagamento = self.parse_pagamento_avulso(pgto_elem)
            operacao.pgto_avls.append(pagamento)
        
        for pcl_elem in element.findall('PclFut'):
            parcela = self.parse_parcelas_futuras(pcl_elem)
            operacao.pcl_fut.append(parcela)
        
        for fat_elem in element.findall('FatFchdAnt'):
            fatura = self.parse_fatura_fechada_anterior(fat_elem)
            operacao.fat_fchd_ant.append(fatura)
        
        for fat_elem in element.findall('FatFchdFut'):
            fatura = self.parse_fatura_fechada_futura(fat_elem)
            operacao.fat_fchd_fut.append(fatura)
        
        for fcht_elem in element.findall('FchtAnt'):
            fechamento = self.parse_fechamento_anterior(fcht_elem)
            operacao.fcht_ant.append(fechamento)
        
        for fcht_elem in element.findall('FchtAtu'):
            fechamento = self.parse_fechamento_atual(fcht_elem)
            operacao.fcht_atu.append(fechamento)
        
        for pcl_elem in element.findall('PclAntCsr'):
            parcela = self.parse_parcela_anterior_consorcio(pcl_elem)
            operacao.pcl_ant_csr.append(parcela)
        
        for pgto_elem in element.findall('PgtoAvlsCsr'):
            pagamento = self.parse_pagamento_avulso_consorcio(pgto_elem)
            operacao.pgto_avls_csr.append(pagamento)
        
        for pcl_elem in element.findall('PclFutCsr'):
            parcela = self.parse_parcelas_futuras_consorcio(pcl_elem)
            operacao.pcl_fut_csr.append(parcela)
        
        return operacao
    
    def parse_cliente(self, element: ET.Element) -> Cliente:
        """Parser para Cliente"""
        cliente = Cliente(
            tip_cli=self._get_text(element, 'TipCli'),
            idfc_cli=self._get_text(element, 'IdfcCli'),
            nm_cli=self._get_text(element, 'NmCli'),
            ntz_rlc=self._get_text(element, 'NtzRlc'),
            cmdo_cli=self._get_text(element, 'CmdoCli')
        )
        
        for opr_elem in element.findall('Opr'):
            operacao = self.parse_operacao(opr_elem)
            cliente.operacoes.append(operacao)
        
        return cliente
    
    def parse_envio_historico_credito(self, element: ET.Element) -> EnvioHistoricoCredito:
        """Parser para EnvioHistoricoCredito (elemento raiz)"""
        envio = EnvioHistoricoCredito(
            cnpj_if=self._get_text(element, 'CnpjIf'),
            cnpj_gbd=self._get_text(element, 'CnpjGbd'),
            nr_rms=self._get_text(element, 'NrRms'),
            seql_rms=self._get_text(element, 'SeqlRms'),
            dt_rms=self._get_text(element, 'DtRms'),
            dt_ref_base_itno=self._get_text(element, 'DtRefBaseItno'),
            fmt_rms=self._get_text(element, 'FmtRms'),
            cd_ocr=self._get_text(element, 'CdOcr')
        )
        
        for cli_elem in element.findall('Cli'):
            cliente = self.parse_cliente(cli_elem)
            envio.Cli.append(cliente)
        
        return envio
    
    
    def parse_xml_file(self, xml_file_path: str) -> EnvioHistoricoCredito:
        """
        Parse completo do arquivo XML
        
        Args:
            xml_file_path: Caminho para o arquivo XML
            
        Returns:
            Objeto EnvioHistoricoCredito com todos os dados parseados
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            ET.ParseError: Se o XML estiver malformado
            ValueError: Se campos obrigatórios estiverem ausentes
        """
        
        if not os.path.exists(xml_file_path):
            raise FileNotFoundError(f"Arquivo XML não encontrado: {xml_file_path}")
        
        try:
            self.logger.info(f"Iniciando parse do arquivo: {xml_file_path}")
            
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            if root.tag not in ['EnvoHstCrd', 'EnvioHistoricoCredito']:
                raise ValueError(f"Elemento raiz inválido: {root.tag}")
            
            resultado = self.parse_envio_historico_credito(root)
            
            self._validate_required_fields(resultado)
            
            self.logger.info(f"Parse concluído com sucesso. Cli processados: {len(resultado.Cli)}")
            
            return resultado
            
        except ET.ParseError as e:
            self.logger.error(f"Erro ao parsear XML: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Erro inesperado durante o parse: {e}")
            raise
    
    def _validate_required_fields(self, envio: EnvioHistoricoCredito):
        """Validar campos obrigatórios"""
        
        required_header = ['cnpj_if', 'cnpj_gbd', 'nr_rms', 'seql_rms', 
                          'dt_rms', 'dt_ref_base_itno', 'fmt_rms']
        
        for field in required_header:
            if not getattr(envio, field):
                raise ValueError(f"Campo obrigatório ausente: {field}")
        
        for cliente in envio.Cli:
            if not all([cliente.tip_cli, cliente.idfc_cli, cliente.nm_cli]):
                raise ValueError(f"Campos obrigatórios ausentes no cliente: {cliente.idfc_cli}")