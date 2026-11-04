use mixeff_rs::formula::{Formula,FixedTerm,RandomTerm,GroupingFactor,RandomCovariance};
use mixeff_rs::model::{LinearMixedModel, data::DataFrame};
use mixeff_rs::model::traits::MixedModelFit;
#[no_mangle]
pub extern "C" fn ranking_probe_mixeff() -> f64 {
 std::panic::catch_unwind(|| {
 let mut d=DataFrame::new();
 let x:Vec<f64>=(0..48).map(|i| ((i%8) as f64)-3.5).collect();
 let y:Vec<f64>=(0..48).map(|i| 2.0 + 0.75*x[i]+0.2*(i/8) as f64+0.1*((i%3) as f64)).collect();
 d.add_numeric("y",y).unwrap(); d.add_numeric("x",x).unwrap();
 d.add_categorical("g",(0..48).map(|i|format!("g{}",i/8)).collect()).unwrap();
 let formula=Formula{response:"y".into(),fixed_terms:vec![FixedTerm::Intercept,FixedTerm::Column("x".into())],random_terms:vec![RandomTerm{terms:vec![FixedTerm::Intercept],grouping:GroupingFactor::Single("g".into()),zerocorr:false,covariance:RandomCovariance::Full,source:None}],derived:vec![]};
 let mut m=LinearMixedModel::new(formula,&d,None).unwrap(); m.fit(true).unwrap();
 m.fixef()[1]
 }).unwrap_or(f64::NAN)
}
