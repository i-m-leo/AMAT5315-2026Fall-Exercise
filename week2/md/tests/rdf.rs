use md::fluid::{radial_distribution, RunMetadata, TrajectoryFrame};

#[test]
fn rdf_counts_a_pair_across_the_periodic_boundary() {
    let metadata = RunMetadata::for_test(2, [10.0, 10.0], 0.02);
    let frame = TrajectoryFrame {
        step: 1,
        t: 0.01,
        pos: vec![[0.2, 5.0], [9.8, 5.0]],
        vel: vec![[0.0; 2]; 2],
        e_pot: 0.0,
        e_kin: 0.0,
    };
    let rdf = radial_distribution(&[frame], &metadata, 10).unwrap();
    assert!(rdf.values[0] > 0.0);
    assert_eq!(rdf.values.len(), 10);
}
