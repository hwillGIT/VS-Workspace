# This is the main application file for the Webtester backend.
# It sets up a basic Flask web server.

# This is the main application file for the Webtester backend.
# It sets up a basic Flask web server and includes logic for database initialization
# based on a configurable schema.

import yaml # Import the PyYAML library for reading YAML configuration files.
import sqlite3 # Import the sqlite3 library for interacting with the SQLite database.
from flask import Flask, jsonify, g # Import Flask, jsonify, and g for building the web application and managing application context.
import os # Import the os module for interacting with the operating system, like checking file paths.

# Define the path to the database file.
# This path is relative to the working directory inside the Docker container.
DATABASE = '/app/data/webtester.db'
# Define the path to the schema configuration file.
# This path is relative to the working directory inside the Docker container.
SCHEMA_CONFIG = '/app/config/schema.yaml'

# Create a Flask application instance.
app = Flask(__name__)

# Function to get a database connection.
def get_db():
  # Check if a database connection already exists in the application context.
  db = getattr(g, '_database', None)
  if db is None:
    # If no connection exists, create a new one to the specified database file.
    db = g._database = sqlite3.connect(DATABASE)
    # Configure the connection to return rows as sqlite3.Row objects, which behave like dictionaries.
    db.row_factory = sqlite3.Row
  return db

# Function to close the database connection at the end of a request.
@app.teardown_appcontext
def close_db(error):
  # Get the database connection from the application context.
  db = getattr(g, '_database', None)
  if db is not None:
    # If a connection exists, close it.
    db.close()

# Function to initialize the database schema.
def init_db():
  # Get a database connection.
  db = get_db()
  # Open the schema configuration file.
  with open(SCHEMA_CONFIG, 'r') as f:
    # Load the schema configuration from the YAML file.
    schema = yaml.safe_load(f)

  # Iterate over the tables defined in the schema.
  for table_name, table_info in schema.get('tables', {}).items():
    # Construct the CREATE TABLE statement.
    columns = []
    for column in table_info.get('columns', []):
      column_definition = f"{column['name']} {column['type']}"
      if column.get('primary_key'):
        column_definition += " PRIMARY KEY"
      if column.get('auto_increment'):
        column_definition += " AUTOINCREMENT"
      columns.append(column_definition)

    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)});"
    # Execute the CREATE TABLE statement.
    db.execute(create_table_sql)

    # Add foreign key constraints if defined.
    for column in table_info.get('columns', []):
      if column.get('foreign_key'):
        fk_info = column['foreign_key']
        add_fk_sql = f"ALTER TABLE {table_name} ADD CONSTRAINT fk_{table_name}_{column['name']} FOREIGN KEY ({column['name']}) REFERENCES {fk_info['table']}({fk_info['column']});"
        # Execute the ALTER TABLE statement to add the foreign key.
        # Note: ALTER TABLE ADD CONSTRAINT is not fully supported in older SQLite versions.
        # For broader compatibility, consider dropping and recreating the table or using a different database.
        try:
            db.execute(add_fk_sql)
        except sqlite3.OperationalError as e:
            # Handle cases where ALTER TABLE ADD CONSTRAINT is not supported or the constraint already exists.
            print(f"Could not add foreign key constraint for {table_name}.{column['name']}: {e}")


  # Commit the changes to the database.
  db.commit()

# Command-line command to initialize the database.
# This can be run using 'flask --app app init-db'.
@app.cli.command('init-db')
def init_db_command():
  # Call the function to initialize the database.
  init_db()
  # Print a success message to the console.
  print('Initialized the database.')

# Define a simple route for the root URL.
@app.route('/')
def index():
  # This function handles requests to the root URL.
  # It returns a JSON response.
  return jsonify({"message": "Welcome to the Webtester Backend!"})

# This block allows the script to be run directly.
# It starts the Flask development server.
if __name__ == '__main__':
  # Ensure the data directory exists before initializing the database.
  # This is important when running locally without Docker volumes.
  if not os.path.exists('/app/data'):
      os.makedirs('/app/data')
  # Initialize the database when the application starts.
  init_db()
  # Run the Flask application in debug mode.
  # Debug mode provides helpful error pages and automatically reloads the server on code changes.
  app.run(debug=True)

# Note: This Flask application now includes database initialization logic
# based on the schema defined in schema.yaml.
# The database file will be created in the /app/data directory, which is
# expected to be a mounted volume in the Docker setup.
# Remember to add extensive comments explaining the code as per the documentation standard.
# The 'g' object is used here to store the database connection in the application context.
# This requires importing 'g' from flask, which will be added in the next step.