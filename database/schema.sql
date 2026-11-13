-- CarVision database schema (MySQL 8+ / MariaDB 10.5+)
-- Run with:  python -m database.init_db      (or paste into the mysql client)
--
-- The `vehicles` table holds the columns CarVision understands. Every column except
-- brand, model, year and selling_price is NULLable, because your CSV may not have all of
-- them. Any CSV column CarVision does not recognise is kept, untouched, in `extra_json`.
-- NOTE: `condition` is a reserved word in MySQL, so that field is called vehicle_condition.

CREATE DATABASE IF NOT EXISTS carvision_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE carvision_db;

CREATE TABLE IF NOT EXISTS vehicles (
  id                INT UNSIGNED NOT NULL AUTO_INCREMENT,
  brand             VARCHAR(100)  NOT NULL,
  model             VARCHAR(150)  NOT NULL,
  variant           VARCHAR(200)  NULL,
  year              SMALLINT UNSIGNED NOT NULL,
  fuel_type         VARCHAR(50)   NULL,
  transmission      VARCHAR(50)   NULL,
  km_driven         INT UNSIGNED  NULL,
  engine_cc         DECIMAL(8,1)  NULL,
  mileage           DECIMAL(7,2)  NULL,
  owner_count       TINYINT UNSIGNED NULL,
  location          VARCHAR(120)  NULL,
  vehicle_condition VARCHAR(50)   NULL,
  selling_price     DECIMAL(14,2) NOT NULL,
  original_price    DECIMAL(14,2) NULL,      -- only if the dataset has a new/ex-showroom price
  image_path        VARCHAR(500)  NULL,      -- only if the dataset links rows to images
  extra_json        JSON          NULL,      -- all other CSV columns
  created_at        TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_brand (brand),
  KEY idx_brand_model (brand, model),
  KEY idx_brand_model_variant (brand, model, variant),
  KEY idx_bmvy (brand, model, variant, year)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS predictions (
  id                INT UNSIGNED NOT NULL AUTO_INCREMENT,
  session_id        VARCHAR(64)   NOT NULL,  -- anonymous browser session; history is private per session
  brand             VARCHAR(100)  NOT NULL,
  model             VARCHAR(150)  NOT NULL,
  variant           VARCHAR(200)  NULL,
  year              SMALLINT UNSIGNED NOT NULL,
  km_driven         INT UNSIGNED  NULL,
  fuel_type         VARCHAR(50)   NULL,
  transmission      VARCHAR(50)   NULL,
  owner_count       TINYINT UNSIGNED NULL,
  engine_cc         DECIMAL(8,1)  NULL,
  mileage           DECIMAL(7,2)  NULL,
  location          VARCHAR(120)  NULL,
  vehicle_condition VARCHAR(50)   NULL,
  predicted_price   DECIMAL(14,2) NOT NULL,
  lower_price       DECIMAL(14,2) NOT NULL,
  upper_price       DECIMAL(14,2) NOT NULL,
  image_path        VARCHAR(255)  NULL,
  result_json       JSON          NOT NULL,  -- full result as shown to the user (stable even if the model is retrained)
  created_at        TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_session_created (session_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
