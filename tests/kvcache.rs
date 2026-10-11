use std::error::Error;

use infer::kvstate;
use models::kvcache;
use tensor::{shape, tensor};

fn row(values: &[f32]) -> Result<tensor<f32>, Box<dyn Error>> {
    Ok(tensor::new(
        shape::new(vec![1, values.len()])?,
        values.to_vec(),
    )?)
}

#[test]
fn appendall_commits_every_layer_together() -> Result<(), Box<dyn Error>> {
    let mut cache = kvstate::new(2, 2, 3)?;
    let key = row(&[1.0, 2.0])?;
    let value = row(&[3.0, 4.0])?;

    cache.appendall(&key, &value)?;

    assert_eq!(cache.getlen(), 1);
    for layer in 0..cache.getlayers() {
        assert_eq!(cache.keys(layer).map(|t| t.getshape().getdims()), Some(&[1, 2][..]));
        assert_eq!(cache.values(layer).map(|t| t.getshape().getdims()), Some(&[1, 2][..]));
    }
    Ok(())
}

#[test]
fn appendall_rejects_diverged_layers_without_partial_mutation(
) -> Result<(), Box<dyn Error>> {
    let mut cache = kvstate::new(2, 2, 3)?;
    let key = row(&[1.0, 2.0])?;
    let value = row(&[3.0, 4.0])?;
    cache.append(0, &key, &value)?;
    cache.append(0, &key, &value)?;

    assert!(cache.appendall(&key, &value).is_err());
    assert_eq!(cache.getlen(), 2);
    assert_eq!(cache.keys(1), None);
    assert_eq!(cache.values(1), None);
    Ok(())
}

#[test]
fn appendall_rejects_capacity_before_mutating_any_layer(
) -> Result<(), Box<dyn Error>> {
    let mut cache = kvstate::new(2, 2, 1)?;
    let key = row(&[1.0, 2.0])?;
    let value = row(&[3.0, 4.0])?;

    cache.appendall(&key, &value)?;
    let keys_before: Vec<Option<Vec<f32>>> = (0..cache.getlayers())
        .map(|layer| cache.keys(layer).map(|value| value.getdata().to_vec()))
        .collect();
    let values_before: Vec<Option<Vec<f32>>> = (0..cache.getlayers())
        .map(|layer| cache.values(layer).map(|value| value.getdata().to_vec()))
        .collect();

    assert!(cache.appendall(&key, &value).is_err());
    assert_eq!(cache.getlen(), 1);
    for layer in 0..cache.getlayers() {
        assert_eq!(cache.keys(layer).map(|value| value.getdata().to_vec()), keys_before[layer]);
        assert_eq!(cache.values(layer).map(|value| value.getdata().to_vec()), values_before[layer]);
    }
    Ok(())
}
