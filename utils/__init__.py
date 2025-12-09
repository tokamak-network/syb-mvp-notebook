import importlib

from .utils import (
    generate_random_eth_address,
    generate_mul_eth_addresses,
    generate_alphabetical_names,
    format_address,
    get_transaction_type_name,
    format_transaction_display,
    TransactionSimulator,
)

__all__ = [
    "generate_random_eth_address",
    "generate_mul_eth_addresses",
    "generate_alphabetical_names",
    "create_random_mvp_network",
    "format_address",
    "get_transaction_type_name",
    "format_transaction_display",
    "TransactionSimulator",
    "build_network_status_html",
    "build_network_figure",
    "generate_random_graph",
    "plot_graph_with_scores",
    "print_score_comparison",
    "plot_score_barplot",
    "show_network_status",
    "show_network_graph",
    "plot_graph_evolution_with_scores",
    "display_graph_state",
]

_LAZY_MODULE_EXPORTS = {
    ".plot_utils": [
        "build_network_status_html",
        "build_network_figure",
        "generate_random_graph",
        "plot_graph_with_scores",
        "print_score_comparison",
        "plot_score_barplot",
        "show_network_status",
        "show_network_graph",
    ],
    ".plot_graphs": [
        "plot_graph_evolution_with_scores",
        "display_graph_state",
    ],
    ".random_network": [
        "create_random_mvp_network",
    ],
}

_ATTR_MODULE_MAP = {
    attr: module for module, attrs in _LAZY_MODULE_EXPORTS.items() for attr in attrs
}


def __getattr__(name):
    module = _ATTR_MODULE_MAP.get(name)
    if module is None:
        raise AttributeError(f"module 'utils' has no attribute '{name}'")

    loaded_module = importlib.import_module(module, __name__)
    value = getattr(loaded_module, name)
    globals()[name] = value
    return value

