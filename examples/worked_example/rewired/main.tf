variable "env" {
  type = string
}

variable "az_count" {
  type    = number
  default = 2
}

locals {
  name = "svc-${var.env}"
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags       = { Name = local.name }
}

resource "aws_subnet" "private" {
  count  = var.az_count
  vpc_id = aws_route_table.rt.vpc_id
}

resource "aws_route_table" "rt" {
  vpc_id = aws_vpc.main.id
}

resource "aws_route_table_association" "assoc" {
  subnet_id      = aws_subnet.private[0].id
  route_table_id = aws_route_table.rt.id
}

resource "aws_s3_bucket" "audit" {
  bucket = "audit-logs"
}

module "logging" {
  source = "./modules/logging"
  vpc_id = aws_vpc.main.id
}

output "subnet_ids" {
  value = aws_subnet.private[*].id
}
