"""
  Copyright (C) 2023 tghuy

  This file is part of VN-Law-Advisor.

  VN-Law-Advisor is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  VN-Law-Advisor is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with VN-Law-Advisor.  If not, see <http://www.gnu.org/licenses/>.
"""

import os

from peewee import MySQLDatabase

_DB_USER = os.getenv("LAW_DB_USER", "root")
_DB_PASS = os.getenv("LAW_DB_PASSWORD", "123456789")
_DB_HOST = os.getenv("LAW_DB_HOST", "localhost")
try:
    _DB_PORT = int(os.getenv("LAW_DB_PORT", "3306"))
except (ValueError, TypeError):
    _DB_PORT = 3306
_DB_NAME = os.getenv("LAW_DB_NAME", "law")

DATABASE = f'mysql://{_DB_USER}:{_DB_PASS}@{_DB_HOST}:{_DB_PORT}/{_DB_NAME}'
db = MySQLDatabase(database=_DB_NAME, user=_DB_USER, password=_DB_PASS, host=_DB_HOST, port=_DB_PORT)

