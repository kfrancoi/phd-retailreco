#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Kevin Fran�oisse on 2010-03-31.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

from sqlalchemy import Table, Column, Integer, String, Float, MetaData, ForeignKey, create_engine
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
dbURL = 'sqlite:///model.db'
engine = create_engine(dbURL, echo=True)

def getSession():
	return sessionmaker(engine)()

def closeSession(session):
	session.close()

def getParameter(param):
	session = getSession()
	p = session.query(Parameter).filter(Parameter.label == param).first()
	if p is None:
		p = Parameter(param)
	closeSession(session)
	return p

def getPerformanceLabel(perf):
	session = getSession()
	perf = session.query(Performance).filter(Performance.label == perf).first()
	if perf is None:
		perf = Performance(perf)
	closeSession(session)
	return perf

def printModelPerformance(model):
	session = getSession()
	for result in session.query(Result).filter(Result.model.has(label=model)):
		for perf in result.performance:
			print '%s : %s'%(perf.performance.label, perf.value)
	

class ResultPerf(Base):
	__tablename__ = 'result_performance'
	
	result_id = Column(Integer, ForeignKey('results.id'), primary_key=True)
	performance_id = Column(Integer, ForeignKey('performances.id'), primary_key=True)
	value = Column(Float)
	performance = relationship('Performance', backref='result_performance')
	
	def __init__(self, performance, value):
		self.performance = performance
		self.value = value

class ModelParameters(Base):
	__tablename__ = 'model_parameter'
	
	model_id = Column(Integer, ForeignKey('models.id'), primary_key=True)
	parameter_id = Column(Integer, ForeignKey('parameters.id'), primary_key=True)
	value = Column(Float)
	parameter = relationship('Parameter', backref='model_parameter')
	
	def __init__(self, parameter, value):
		self.parameter = parameter
		self.value = value

class Result(Base):
	__tablename__ = 'results'

	id = Column(Integer, primary_key=True)
	model_id = Column(Integer, ForeignKey('models.id'))
	model = relationship('Model', backref='results')
	performances = relationship(ResultPerf, backref='results')
	
	def __init__(self):
		pass


class Model(Base):
	__tablename__='models'

	id = Column(Integer, primary_key=True)
	label = Column(String)
	parameters = relationship(ModelParameters, backref='models')
	
	def __init__(self, label):
		self.label = label
	
	def __repr__(self):
		return '<Model(%s)>'%self.label
		

class Parameter(Base):
	__tablename__= 'parameters'
	
	id = Column(Integer, primary_key=True)
	label = Column(String, unique=True)
	
	def __init__(self, label):
		self.label = label
		
	def __repr__(self):
		return '<Parameters(%s)>'%self.label

class Performance(Base):
	__tablename__ = 'performances'
	
	id = Column(Integer, primary_key=True)
	label = Column(String, unique=True)
	
	def __init__(self, label):
		self.label = label


Base.metadata.create_all(engine)
