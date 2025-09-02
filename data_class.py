from dataclasses import dataclass, field
from typing import List, Optional
@dataclass
class PagamentoParcelaAnterior:
    """Representa um pagamento de parcela anterior"""
    dt_pgto_pcl_ant: Optional[str] = None  # DataPagamento
    vl_pgto_pcl_ant: Optional[float] = None  # ValorPagamento

@dataclass
class ParcelaAnterior:
    """Representa uma parcela anterior"""
    dt_vnct_pcl_ant: Optional[str] = None  # DataVencimento
    perc_pcl_ant: Optional[str] = None  # PeriodicidadeParcelaAnterior
    vl_pcl_ant: Optional[float] = None  # Valor
    cmdo_pcl_ant: Optional[str] = None  # ComandoParcelaAnterior
    pgto_pcl_ant: List[PagamentoParcelaAnterior] = field(default_factory=list)

@dataclass
class PagamentoAvulso:
    """Representa um pagamento avulso"""
    dt_pgto_avls: Optional[str] = None  # DataPagamento
    vl_pgto_avls: Optional[float] = None  # ValorPagamento
    tip_pgto_avls: Optional[str] = None  # TipoPagamento
    cmdo_pgto_avls: Optional[str] = None  # ComandoPagamentoAvulso

@dataclass
class ParcelasFuturas:
    """Representa parcelas futuras"""
    dt_vnct_prx_pcl: Optional[str] = None  # DataVencimentoProximaParcela
    perc_pcl_fut: Optional[str] = None  # PeriodicidadeParcelasFuturas
    vl_prx_pcl: Optional[float] = None  # ValorProximaParcela
    qt_pcl_vncr: Optional[int] = None  # QuantidadeParcelasAVencer
    qt_pcl_pgr: Optional[int] = None  # QuantidadeParcelasAPagar

@dataclass
class PagamentoFaturaFechadaAnterior:
    """Representa pagamento de fatura fechada anterior"""
    dt_pgto_fat_fchd_ant: Optional[str] = None  # DataPagamento
    vl_pgto_fat_fchd_ant: Optional[float] = None  # ValorPagamento

@dataclass
class FaturaFechadaAnterior:
    """Representa uma fatura fechada anterior"""
    dt_vnct_fat_fchd_ant: Optional[str] = None  # DataVencimento
    vl_ttl_pgr_fat_fchd_ant: Optional[float] = None  # ValorTotalAPagar
    vl_min_pgr_fat_fchd_ant: Optional[float] = None  # ValorMinimoAPagar
    cmdo_fat_fchd_ant: Optional[str] = None  # ComandoFaturaFechadaAnterior
    pgto_fat_fchd_ant: List[PagamentoFaturaFechadaAnterior] = field(default_factory=list)

@dataclass
class FaturaFechadaFutura:
    """Representa uma fatura fechada futura"""
    dt_vnct_fat_fchd_fut: Optional[str] = None  # DataVencimento
    vl_ttl_pgr_fat_fchd_fut: Optional[float] = None  # ValorTotalAPagar
    vl_min_pgr_fat_fchd_fut: Optional[float] = None  # ValorMinimoAPagar
    cmdo_fat_fch_fut: Optional[str] = None  # ComandoFaturaFechadaFutura

@dataclass
class FechamentoAnterior:
    """Representa um fechamento anterior"""
    dt_fcht_ant: Optional[str] = None  # Data
    sdo_utlz_fcht_ant: Optional[float] = None  # SaldoUtilizado
    cmdo_fcht_ant: Optional[str] = None  # ComandoFechamentoAnterior

@dataclass
class FechamentoAtual:
    """Representa um fechamento atual"""
    dt_fcht_atu: Optional[str] = None  # Data
    sdo_utlz_fcht_atu: Optional[float] = None  # SaldoUtilizado
    cmdo_fcht_atu: Optional[str] = None  # ComandoFechamentoAtual

@dataclass
class PagamentoParcelaAnteriorConsorcio:
    """Representa pagamento de parcela anterior do consórcio"""
    dt_pgto_pcl_ant_csr: Optional[str] = None  # DataPagamento
    vl_pgto_pcl_ant_csr: Optional[float] = None  # ValorPagamento

@dataclass
class ParcelaAnteriorConsorcio:
    """Representa parcela anterior do consórcio"""
    dt_vnct_pcl_ant_csr: Optional[str] = None  # DataVencimento
    perc_pcl_ant_csr: Optional[str] = None  # PeriodicidadeParcelaAnterior
    vl_pcl_ant_csr: Optional[float] = None  # Valor
    cmdo_pcl_ant_csr: Optional[str] = None  # ComandoParcelaAnteriorConsorcio
    pgto_pcl_ant_csr: List[PagamentoParcelaAnteriorConsorcio] = field(default_factory=list)

@dataclass
class PagamentoAvulsoConsorcio:
    """Representa pagamento avulso do consórcio"""
    dt_pgto_avls_csr: Optional[str] = None  # DataPagamento
    vl_pgto_avls_csr: Optional[float] = None  # ValorPagamento
    tip_pgto_avls_csr: Optional[str] = None  # TipoPagamento
    cmdo_pgto_avls_csr: Optional[str] = None  # ComandoPagamentoAvulsoConsorcio

@dataclass
class ParcelasFuturasConsorcio:
    """Representa parcelas futuras do consórcio"""
    dt_vnct_prx_pcl_csr: Optional[str] = None  # DataVencimentoProximaParcela
    perc_pcl_fut_csr: Optional[str] = None  # PeriodicidadeParcelasFuturasConsorcio
    vl_prx_pcl_csr: Optional[float] = None  # ValorProximaParcela
    qt_pcl_vncr_csr: Optional[int] = None  # QuantidadeParcelasAVencer
    qt_pcl_pgr_ct_csr: Optional[int] = None  # QuantidadeParcelasAPagarCota

@dataclass
class DetalheOperacao:
    """Representa o detalhe de uma operação"""
    in_pre_fix: Optional[str] = None  # IndicadorPreFixado
    dt_vnct_ult_pcl: Optional[str] = None  # DataVencimentoUltimaParcela
    vl_ctrd_fut: Optional[float] = None  # ValorContratadoFuturo
    qt_pcl: Optional[int] = None  # QuantidadeParcelas
    nr_plst_crt: Optional[str] = None  # NúmeroPlásticoCartão
    sdo_dvdr: Optional[float] = None  # SaldoDevedor
    dt_vnct_opr: Optional[str] = None  # DataVencimento
    nr_gr: Optional[str] = None  # Grupo
    cd_ct: Optional[str] = None  # Cota
    seq_ct: Optional[str] = None  # Sequencia
    sit_ct: Optional[str] = None  # SituacaoCota
    vl_obgc: Optional[float] = None  # ValorContratadoObrigacao
    dt_vnct_ult_pcl_csr: Optional[str] = None  # DataVencimentoUltimaParcelaCsr
    qt_pcl_ct_csr: Optional[int] = None  # QuantidadeParcelasContaCsr
    sdo_dvdr_ct_csr: Optional[float] = None  # SaldoDevedorContaCsr
    dt_cont_ct_csr: Optional[str] = None  # DataContemplacaoCota

@dataclass
class Operacao:
    """Representa uma operação de crédito"""
    nr_unco: Optional[str] = None  # NumeroUnico
    prf_ag: Optional[str] = None  # PrefixoAgencia
    nr_ctr: Optional[str] = None  # NumeroContrato
    dt_ctrc: Optional[str] = None  # DataContratacao
    cd_mdld: Optional[str] = None  # Modalidade
    dt_aprc: Optional[str] = None  # DataApuracao
    cnpj_ctrc: Optional[str] = None  # CnpjContratacao
    cmdo_opr: Optional[str] = None  # Comando
    idfc_cli_ctrt: Optional[str] = None  # IdentificacaoClienteContrato
    tip_rst: Optional[str] = None  # TipoRestricao
    
    # Sub-elementos
    det_opr: Optional[DetalheOperacao] = None
    pcl_ant: List[ParcelaAnterior] = field(default_factory=list)
    pgto_avls: List[PagamentoAvulso] = field(default_factory=list)
    pcl_fut: List[ParcelasFuturas] = field(default_factory=list)
    fat_fchd_ant: List[FaturaFechadaAnterior] = field(default_factory=list)
    fat_fchd_fut: List[FaturaFechadaFutura] = field(default_factory=list)
    fcht_ant: List[FechamentoAnterior] = field(default_factory=list)
    fcht_atu: List[FechamentoAtual] = field(default_factory=list)
    pcl_ant_csr: List[ParcelaAnteriorConsorcio] = field(default_factory=list)
    pgto_avls_csr: List[PagamentoAvulsoConsorcio] = field(default_factory=list)
    pcl_fut_csr: List[ParcelasFuturasConsorcio] = field(default_factory=list)

@dataclass
class Cliente:
    """Representa um cliente"""
    tip_cli: str  # Tipo (obrigatório)
    idfc_cli: str  # Identificacao (obrigatório)
    nm_cli: str  # Nome (obrigatório)
    ntz_rlc: Optional[str] = None  # NaturezaRelacao
    cmdo_cli: Optional[str] = None  # ComandoCliente
    
    # Sub-elementos
    operacoes: List[Operacao] = field(default_factory=list)

@dataclass
class EnvioHistoricoCredito:
    """Representa o elemento raiz do XML"""
    cnpj_if: str  # Cnpj da Fonte (obrigatório)
    cnpj_gbd: str  # Cnpj do GBD (obrigatório)
    nr_rms: str  # NumeroRemessa (obrigatório)
    seql_rms: str  # SequencialRemessa (obrigatório)
    dt_rms: str  # DataRemessa (obrigatório)
    dt_ref_base_itno: str  # DataReferenciaBaseInterna (obrigatório)
    fmt_rms: str  # FormatoRemessa (obrigatório)
    cd_ocr: Optional[str] = None  # Código de Ocorrência
    
    # Sub-elementos
    Cli: List[Cliente] = field(default_factory=list)