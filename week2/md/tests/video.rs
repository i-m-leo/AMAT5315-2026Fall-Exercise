use md::fluid::{radial_distribution, render_video_frame, RunMetadata, TrajectoryFrame};

#[test]
fn rendered_video_frame_has_the_requested_rgb_size() {
    let metadata = RunMetadata::for_test(2, [10.0, 10.0], 0.02);
    let frame = TrajectoryFrame {
        step: 1,
        t: 0.01,
        pos: vec![[2.0, 5.0], [8.0, 5.0]],
        vel: vec![[0.1, 0.0], [-0.1, 0.0]],
        e_pot: 0.0,
        e_kin: 0.01,
    };
    let rdf = radial_distribution(std::slice::from_ref(&frame), &metadata, 10).unwrap();
    let rgb = render_video_frame(&frame, &rdf, &metadata, [640, 360]).unwrap();
    assert_eq!(rgb.len(), 640 * 360 * 3);
    assert!(rgb.windows(2).any(|channels| channels[0] != channels[1]));
}
