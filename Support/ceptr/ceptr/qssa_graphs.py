"""Graph utilities for QSSA."""

import pathlib

import matplotlib.pyplot as plt
import networkx as nx

import ceptr.utilities as cu


def directed_graph(mechanism, qssa_species):
    """Graph of links between QSSA species."""
    links = []
    for reaction in mechanism.reactions():
        # Identify QSS species involved
        reac_qssa_involved = cu.intersection(
            list(reaction.reactants.keys()), qssa_species
        )
        prod_qssa_involved = cu.intersection(
            list(reaction.products.keys()), qssa_species
        )
        if (prod_qssa_involved) and (reac_qssa_involved):
            for prod in prod_qssa_involved:
                for reac in reac_qssa_involved:
                    links.append((prod, reac))
            if reaction.reversible:
                for reac in reac_qssa_involved:
                    for prod in prod_qssa_involved:
                        links.append((reac, prod))
        if len(reac_qssa_involved) > 1:
            for reac1 in reac_qssa_involved:
                for reac2 in reac_qssa_involved:
                    if not reac1 == reac2:
                        links.append((reac1, reac2))

        if len(prod_qssa_involved) > 1:
            for prod1 in prod_qssa_involved:
                for prod2 in prod_qssa_involved:
                    if not prod1 == prod2:
                        links.append((prod1, prod2))

    return links


def plot_directed_graph(mechanism, links):
    """Plot directed graph of QSSA."""
    mechpath = pathlib.Path(mechanism.source)
    fname = mechpath.parents[0] / "qssa_directed_graph.png"
    # A figure of its own: networkx draws on the current one
    fig = plt.figure()

    # Make Graph
    graph = nx.DiGraph()
    graph.add_edges_from(links)

    # Specify the edges you want here
    node_size = []
    labels = {}
    # define color map. user_node = red, book_nodes = green
    node_color = []
    for node in graph.nodes():
        node_size.append(0)
        labels[node] = node
        node_color.append("black")

    black_edges = [edge for edge in graph.edges()]

    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color)
    nx.draw_networkx_labels(graph, pos, labels, font_size=14, font_color="k")
    nx.draw_networkx_edges(graph, pos, edgelist=black_edges, arrows=True)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close(fig)


def qssa_quadratic_graph(mechanism, reaction_info, qssa_species):
    """Graph of quadratic couplings."""
    links = []
    quadratic_reaction = []
    problematic_qssa_species = []
    for orig_idx, _ in reaction_info.idxmap.items():
        reaction = mechanism.reaction(orig_idx)

        # Identify QSS species involved
        reac_qssa_involved = cu.intersection(
            list(reaction.reactants.keys()), qssa_species
        )
        prod_qssa_involved = cu.intersection(
            list(reaction.products.keys()), qssa_species
        )
        reac_sum_coeff = sum(
            v for k, v in reaction.reactants.items() if k in reac_qssa_involved
        )
        prod_sum_coeff = sum(
            v for k, v in reaction.products.items() if k in prod_qssa_involved
        )
        if (prod_qssa_involved) and (reac_qssa_involved):
            for prod in prod_qssa_involved:
                links.append((prod, "F" + str(orig_idx)))
                for reac in reac_qssa_involved:
                    links.append(("F" + str(orig_idx), reac))
            if reaction.reversible:
                for reac in reac_qssa_involved:
                    links.append((reac, "R" + str(orig_idx)))
                    for prod in prod_qssa_involved:
                        links.append(("R" + str(orig_idx), prod))
        if (not prod_qssa_involved) and (reac_qssa_involved) and reac_sum_coeff > 1:
            for reac in reac_qssa_involved:
                links.append(("F" + str(orig_idx), reac))
        if (
            (prod_qssa_involved)
            and (not reac_qssa_involved)
            and prod_sum_coeff > 1
            and reaction.reversible
        ):
            for prod in prod_qssa_involved:
                links.append(("R" + str(orig_idx), prod))

        if reac_sum_coeff > 1:
            # This is a quadratic coupling
            quadratic_reaction.append("F" + str(orig_idx))
            problematic_qssa_species += reac_qssa_involved
            problematic_qssa_species += prod_qssa_involved

        if reaction.reversible:
            if prod_sum_coeff > 1:
                quadratic_reaction.append("R" + str(orig_idx))
                problematic_qssa_species += reac_qssa_involved
                problematic_qssa_species += prod_qssa_involved

    return links, quadratic_reaction, problematic_qssa_species


def plot_quadratic_graph(mechanism, links, quadratic_reactions, problematic_species):
    """Plot QSSA quadratic graph."""
    mechpath = pathlib.Path(mechanism.source)
    fname = mechpath.parents[0] / "qssa_quadratic_graph.png"
    # A figure of its own: networkx draws on the current one
    fig = plt.figure()

    # Make Graph
    graph = nx.DiGraph()
    graph.add_edges_from(links)

    # Specify the edges you want here
    labels_normal_species = {}
    labels_problem_species = {}
    # define color map. user_node = red, book_nodes = green
    node_list_normal_species = []
    node_list_problem_species = []
    node_list_normal_reactions = []
    node_list_problem_reactions = []
    for node in graph.nodes():
        if node in quadratic_reactions:
            node_list_problem_reactions.append(node)
        elif node in problematic_species:
            node_list_problem_species.append(node)
            labels_problem_species[node] = node
        elif node.startswith("F") or node.startswith("R"):
            node_list_normal_reactions.append(node)
        else:
            node_list_normal_species.append(node)
            labels_normal_species[node] = node

    black_edges = [edge for edge in graph.edges()]

    # Create a layout
    pos = nx.kamada_kawai_layout(graph)

    # Draw normal species
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=node_list_normal_species,
        node_size=0,
        node_color="b",
    )
    label_options = {"ec": "k", "fc": "white", "alpha": 1}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels_normal_species,
        font_size=14,
        font_color="k",
        bbox=label_options,
    )

    # Draw problem species
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=node_list_problem_species,
        node_size=0,
        node_color="b",
    )
    label_options = {"ec": "k", "fc": "red", "alpha": 1}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels_problem_species,
        font_size=14,
        font_color="k",
        bbox=label_options,
    )

    # Draw normal reactions
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=node_list_normal_reactions,
        node_size=200,
        node_color="w",
        edgecolors="k",
    )

    # Draw problem reactions
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=node_list_problem_reactions,
        node_size=200,
        node_color="r",
    )

    nx.draw_networkx_edges(graph, pos, edgelist=black_edges, arrows=False)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close(fig)
