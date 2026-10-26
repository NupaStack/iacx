"""
Extraction regression tests.

These exist because the measurement tool silently under-extracted dependency
edges across an entire 542-repository corpus. python-hcl2 returns block labels
with their surrounding quotes, so vertices were recorded as '"aws_vpc"."main"'
while the reference resolver searched for 'aws_vpc.main'. Roughly four edges in
five were lost. Every test below asserts a hand-derived expected graph.

Run:  python3 -m pytest tool/tests/test_extraction.py -q
"""
import os, sys, tempfile, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'iacx'))
EX = os.path.join(os.path.dirname(__file__), '..', 'examples', 'worked_example')
import iacx


def build(src):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, 'main.tf'), 'w') as fh:
        fh.write(textwrap.dedent(src))
    nodes, _ = iacx.parse_directory(d, recursive=False)
    return nodes, iacx.build_graph(nodes), iacx.analyse(d, recursive=False)


def test_vertex_ids_are_unquoted():
    nodes, _, _ = build('''
        resource "aws_vpc" "main" { cidr_block = "10.0.0.0/16" }
        variable "env" { type = string }
    ''')
    assert 'aws_vpc.main' in nodes, nodes
    assert 'var.env' in nodes, nodes
    assert not any('"' in n for n in nodes), nodes


def test_parser_marker_is_not_a_vertex():
    nodes, _, _ = build('locals { name = "a" }')
    assert 'local.__is_block__' not in nodes
    assert 'local.name' in nodes


def test_resource_to_resource_edge():
    _, edges, _ = build('''
        resource "aws_vpc" "main" { cidr_block = "10.0.0.0/16" }
        resource "aws_subnet" "a" { vpc_id = aws_vpc.main.id }
    ''')
    assert ('aws_vpc.main', 'aws_subnet.a') in edges, edges


def test_variable_and_local_edges():
    _, edges, _ = build('''
        variable "env" { type = string }
        locals { name = "svc-${var.env}" }
        resource "aws_vpc" "main" { tags = { Name = local.name } }
    ''')
    assert ('var.env', 'local.name') in edges, edges
    assert ('local.name', 'aws_vpc.main') in edges, edges


def test_worked_example_graph_is_exact():
    """An eleven-edge hand-derived fixture (not Listing 1; see test_listing1_* below)."""
    _, edges, m = build('''
        variable "env" { type = string }
        variable "subnet_count" { type = number
                                  default = 2 }
        locals { name = "svc-${var.env}" }
        resource "aws_vpc" "main" {
          cidr_block = "10.0.0.0/16"
          tags       = { Name = local.name }
        }
        resource "aws_subnet" "private" {
          count      = var.subnet_count
          vpc_id     = aws_vpc.main.id
          cidr_block = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
        }
        resource "aws_route_table" "private" { vpc_id = aws_vpc.main.id }
        resource "aws_route_table_association" "private" {
          count          = var.subnet_count
          subnet_id      = aws_subnet.private[count.index].id
          route_table_id = aws_route_table.private.id
        }
        module "logging" {
          source      = "./modules/logging"
          vpc_id      = aws_vpc.main.id
          name_prefix = local.name
        }
        output "subnet_ids" { value = aws_subnet.private[*].id }
    ''')
    expected = {
        ('var.env',               'local.name'),
        ('local.name',            'aws_vpc.main'),
        ('local.name',            'module.logging'),
        ('aws_vpc.main',          'aws_subnet.private'),
        ('aws_vpc.main',          'aws_route_table.private'),
        ('aws_vpc.main',          'module.logging'),
        ('var.subnet_count',      'aws_subnet.private'),
        ('var.subnet_count',      'aws_route_table_association.private'),
        ('aws_subnet.private',    'aws_route_table_association.private'),
        ('aws_route_table.private','aws_route_table_association.private'),
        ('aws_subnet.private',    'output.subnet_ids'),
    }
    assert edges == expected, f"missing {expected - edges}\nextra {edges - expected}"
    assert m['edges'] == 11 and m['D_depth'] == 4, m


def test_declaration_order_does_not_change_the_measures():
    a = '''
        resource "aws_vpc" "main" { cidr_block = "10.0.0.0/16" }
        resource "aws_subnet" "a" { vpc_id = aws_vpc.main.id }
    '''
    b = '''
        resource "aws_subnet" "a" { vpc_id = aws_vpc.main.id }
        resource "aws_vpc" "main" { cidr_block = "10.0.0.0/16" }
    '''
    _, _, ma = build(a)
    _, _, mb = build(b)
    for k in ('D_depth', 'P_propagation', 'X_expression', 'M_modularity',
              'K_crossing', 'IaCX', 'edges'):
        assert ma[k] == mb[k], (k, ma[k], mb[k])


def test_consistent_renaming_does_not_change_the_measures():
    a = '''
        resource "aws_vpc" "main" { cidr_block = "10.0.0.0/16" }
        resource "aws_subnet" "a" { vpc_id = aws_vpc.main.id }
    '''
    b = a.replace('"main"', '"primary"').replace('aws_vpc.main', 'aws_vpc.primary')
    _, _, ma = build(a)
    _, _, mb = build(b)
    for k in ('D_depth', 'P_propagation', 'IaCX', 'edges'):
        assert ma[k] == mb[k], (k, ma[k], mb[k])


def test_dependency_rearrangement_does_change_the_measures():
    """P7' from the paper: rewiring without adding or removing blocks."""
    chain = '''
        resource "aws_vpc" "a"    { cidr_block = "10.0.0.0/16" }
        resource "aws_subnet" "b" { vpc_id = aws_vpc.a.id }
        resource "aws_subnet" "c" { vpc_id = aws_subnet.b.id }
    '''
    star = '''
        resource "aws_vpc" "a"    { cidr_block = "10.0.0.0/16" }
        resource "aws_subnet" "b" { vpc_id = aws_vpc.a.id }
        resource "aws_subnet" "c" { vpc_id = aws_vpc.a.id }
    '''
    _, _, mc = build(chain)
    _, _, ms = build(star)
    assert mc['edges'] == ms['edges'] == 2
    assert mc['D_depth'] == 2 and ms['D_depth'] == 1, (mc['D_depth'], ms['D_depth'])


# ---------------------------------------------------------------------------
# Listing 1 of the paper (examples/worked_example/base/main.tf), with the two
# transformations reported in Table 2. Expected values were derived by hand.
# ---------------------------------------------------------------------------
LISTING1_EDGES = {
    ('var.env',            'local.name'),
    ('local.name',         'aws_vpc.main'),
    ('var.az_count',       'aws_subnet.private'),
    ('aws_vpc.main',       'aws_subnet.private'),
    ('aws_vpc.main',       'aws_route_table.rt'),
    ('aws_vpc.main',       'module.logging'),
    ('aws_subnet.private', 'aws_route_table_association.assoc'),
    ('aws_route_table.rt', 'aws_route_table_association.assoc'),
    ('aws_subnet.private', 'output.subnet_ids'),
}
LISTING1_VECTOR = dict(nodes=10, edges=9, D_depth=4, P_propagation=18, X_expression=1.5,
                       M_modularity=1, K_crossing=1, IaCX=25.5)


def measure_dir(d):
    nodes, _ = iacx.parse_directory(d, recursive=False)
    return nodes, iacx.build_graph(nodes), iacx.analyse(d, recursive=False)


def test_listing1_graph_and_vector_are_exact():
    nodes, edges, m = measure_dir(os.path.join(EX, 'base'))
    assert edges == LISTING1_EDGES, f"missing {LISTING1_EDGES - edges}\nextra {edges - LISTING1_EDGES}"
    for k, v in LISTING1_VECTOR.items():
        assert m[k] == v, (k, m[k], v)
    assert 'aws_s3_bucket.audit' in nodes          # the independent resource is a vertex
    assert m['cyclic'] is False


def test_listing1_permuted_declarations_are_identical():
    _, eb, mb = measure_dir(os.path.join(EX, 'base'))
    _, er, mr = measure_dir(os.path.join(EX, 'reordered'))
    assert er == eb
    for k in LISTING1_VECTOR:
        assert mr[k] == mb[k], (k, mr[k], mb[k])


def test_listing1_one_redirected_reference_changes_depth_only():
    _, e, m = measure_dir(os.path.join(EX, 'rewired'))
    assert m['nodes'] == 10 and m['edges'] == 9
    assert m['D_depth'] == 5 and m['P_propagation'] == 18 and m['IaCX'] == 26.5, m
    assert ('aws_route_table.rt', 'aws_subnet.private') in e


def test_splitting_or_merging_locals_blocks_does_not_change_the_graph():
    one = """
        variable "env" { type = string }
        locals {
          name = "svc-${var.env}"
          tags = { Name = local.name }
        }
        resource "aws_vpc" "main" { tags = local.tags }
    """
    two = """
        variable "env" { type = string }
        locals { name = "svc-${var.env}" }
        locals { tags = { Name = local.name } }
        resource "aws_vpc" "main" { tags = local.tags }
    """
    n1, e1, m1 = build(one)
    n2, e2, m2 = build(two)
    assert set(n1) == set(n2) == {'var.env', 'local.name', 'local.tags', 'aws_vpc.main'}
    assert e1 == e2 == {('var.env', 'local.name'), ('local.name', 'local.tags'), ('local.tags', 'aws_vpc.main')}
    assert m1['D_depth'] == m2['D_depth'] == 3


def test_unresolved_references_create_no_edge():
    _, edges, m = build("""
        resource "aws_subnet" "a" {
          vpc_id = aws_vpc.elsewhere.id
          zone   = data.aws_availability_zones.missing.names[0]
          role   = module.not_here.arn
        }
    """)
    assert edges == set(), edges
    assert m['edges'] == 0 and m['D_depth'] == 0


def test_iterator_and_meta_references_are_not_edges():
    _, edges, _ = build("""
        variable "replica" { type = list(any) }
        variable "names"   { type = set(string) }
        resource "aws_secretsmanager_secret" "main" {
          count = 2
          name  = "s-${count.index}"
          dynamic "replica" {
            for_each = var.replica
            content { region = replica.value }
          }
        }
        resource "aws_s3_bucket" "b" {
          for_each = var.names
          bucket   = each.value
          path     = path.module
        }
    """)
    assert edges == {('var.replica', 'aws_secretsmanager_secret.main'),
                     ('var.names', 'aws_s3_bucket.b')}, edges


def test_cycles_are_reported_not_approximated():
    _, edges, m = build("""
        locals {
          a = local.b
          b = local.a
        }
        resource "aws_vpc" "v" { tags = { n = local.a } }
    """)
    assert ('local.a', 'local.b') in edges and ('local.b', 'local.a') in edges
    assert m['cyclic'] is True and m['D_depth'] is None and m['IaCX'] is None


def test_literal_text_is_not_scanned_but_interpolations_are():
    """Version 1.0.0: quoted literals and heredocs contribute only their ${...} interpolations."""
    _, edges, m = build('''
        variable "env" {
          description = "Is this for aws_vpc.main? Use var.env or local.name for each zone."
        }
        resource "aws_vpc" "main" {
          cidr_block = "10.0.0.0/16"
        }
        resource "aws_subnet" "a" {
          tags = { Note = "for x in list : var.env ? a : b", Ref = "id-${aws_vpc.main.id}" }
          policy = <<EOT
        count = length([for s in var.env : s])  aws_vpc.main
        EOT
        }
    ''')
    assert edges == {('aws_vpc.main', 'aws_subnet.a')}, edges
    assert m['X_expression'] == 0, m['X_expression']
