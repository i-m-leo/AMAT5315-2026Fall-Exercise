//! A minimal writer for version-1.0 `.npy` files.
//!
//! numpy stores a little-endian dtype string, the shape, and a row-major data
//! block after a plain-text dictionary. The dictionary is padded with spaces
//! so that the data starts on a 64-byte boundary.

/// One `.npy` array: its dtype string (`<f8` or `<f4`) and shape.
pub struct Npy<'a, T> {
    pub descr: &'static str,
    pub shape: &'a [usize],
    pub data: &'a [T],
}

fn header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let mut dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': (");
    for (i, dim) in shape.iter().enumerate() {
        if i > 0 {
            dict.push_str(", ");
        }
        dict.push_str(&dim.to_string());
    }
    // numpy writes a trailing comma for one-element tuples.
    if shape.len() == 1 {
        dict.push(',');
    }
    dict.push_str("), }");
    // numpy reserves room for a growing first axis before padding.
    let growth = 21usize.saturating_sub(shape.first().map_or(1, |d| d.to_string().len()));
    dict.push_str(&" ".repeat(growth));
    // 10 magic bytes + 2-byte length + dict + padding + newline is a
    // multiple of 64.
    let hlen = dict.len() + 1;
    let pad = (64 - (10 + hlen) % 64) % 64;
    let mut out = Vec::with_capacity(10 + hlen);
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&((hlen + pad) as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat(b' ').take(pad));
    out.push(b'\n');
    out
}

/// Write `array` to `path` in numpy version-1.0 format.
pub fn write_f64(path: &std::path::Path, shape: &[usize], data: &[f64]) -> Result<(), String> {
    write(path, "<f8", shape, data.iter().flat_map(|v| v.to_le_bytes()).collect())
}

/// Write a float32 array to `path` in numpy version-1.0 format.
pub fn write_f32(path: &std::path::Path, shape: &[usize], data: &[f32]) -> Result<(), String> {
    write(path, "<f4", shape, data.iter().flat_map(|v| v.to_le_bytes()).collect())
}

fn write(path: &std::path::Path, descr: &str, shape: &[usize], bytes: Vec<u8>) -> Result<(), String> {
    let mut out = header(descr, shape);
    out.extend_from_slice(&bytes);
    std::fs::write(path, out).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Read a float64 `.npy` file, checking its dtype and shape.
pub fn read_f64(path: &std::path::Path, expected_shape: &[usize]) -> Result<Vec<f64>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a .npy file", path.display()));
    }
    let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let header = std::str::from_utf8(&raw[10..10 + hlen - 1])
        .map_err(|_| format!("{} has a non-utf8 header", path.display()))?;
    if !header.contains("'descr': '<f8'") {
        return Err(format!("{} is not a little-endian float64 array", path.display()));
    }
    if header.contains("'fortran_order': True") {
        return Err(format!("{} is in Fortran order", path.display()));
    }
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .ok_or_else(|| format!("{} has no shape", path.display()))?;
    let shape: Vec<usize> = shape_text
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect();
    if shape != expected_shape {
        return Err(format!(
            "{} has shape {shape:?}, expected {expected_shape:?}",
            path.display()
        ));
    }
    let body = &raw[10 + hlen..];
    let data: Vec<f64> = body
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if data.len() != expected_shape.iter().product::<usize>() {
        return Err(format!("{} has a truncated data block", path.display()));
    }
    Ok(data)
}
